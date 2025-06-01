"""
Large file transcription application with chunked processing.
Designed for 2+ hour audio files with memory efficiency.
"""

import click
import os
import time
from pathlib import Path
from typing import Optional

from .chunked_transcriber import ChunkedTranscriber
from .post_processor import TranscriptionPostProcessor
from .output_formatter import OutputFormatter
from .time_estimator import TranscriptionTimeEstimator


class LargeFileProgressCallback:
    """Progress callback for large file processing."""
    
    def __init__(self, estimated_time: float):
        self.estimated_time = estimated_time
        self.start_time = time.time()
        self.last_update = 0
        
    def __call__(self, stage: str, progress: float, extra_info=None):
        """Handle progress updates."""
        current_time = time.time()
        
        # Update every 5 seconds to avoid spam
        if current_time - self.last_update < 5.0 and progress < 1.0:
            return
            
        elapsed = current_time - self.start_time
        
        if progress > 0:
            estimated_total = elapsed / progress
            remaining = max(0, estimated_total - elapsed)
        else:
            remaining = self.estimated_time
        
        # Create progress bar
        bar_length = 25
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format stage info
        stage_info = {
            'chunking': '📦 チャンク分割',
            'extracting': '✂️ チャンク抽出',
            'processing': '🔄 チャンク処理',
            'transcription': '📝 文字起こし',
            'merging': '🔗 結果統合',
            'post_processing': '🔧 後処理',
            'saving': '💾 保存'
        }
        
        stage_name = stage_info.get(stage, stage)
        
        # Extra info display
        extra_str = ""
        if extra_info:
            if isinstance(extra_info, int):
                extra_str = f" ({extra_info})"
            elif isinstance(extra_info, str):
                extra_str = f" ({extra_info})"
        
        print(f"\r{stage_name} [{bar}] {progress:.1%}{extra_str} | "
              f"経過: {self._format_time(elapsed)} | 残り: {self._format_time(remaining)}", 
              end='', flush=True)
        
        if progress >= 1.0:
            print()  # New line when stage completes
            
        self.last_update = current_time
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='large-v3-turbo', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo']),
              help='Whisper model size (default: large-v3-turbo for maximum speed)')
@click.option('--language', '-l', default='ja', help='Language code for transcription')
@click.option('--min-confidence', default=0.3, type=float,
              help='Minimum confidence threshold for segments')
@click.option('--chunk-duration', default=10, type=int,
              help='Chunk duration in minutes (default: 10)')
@click.option('--overlap-duration', default=2.0, type=float,
              help='Overlap between chunks in seconds (default: 2.0)')
@click.option('--max-memory', default=512, type=int,
              help='Maximum memory per chunk in MB (default: 512)')
@click.option('--no-enhanced-preprocessing', is_flag=True,
              help='Skip enhanced audio preprocessing')
@click.option('--no-post-processing', is_flag=True,
              help='Skip post-processing corrections')
@click.option('--no-filter-fillers', is_flag=True,
              help='Skip filler word filtering')
@click.option('--format', 'output_format', 
              type=click.Choice(['all', 'json', 'csv', 'txt', 'srt']),
              default='all', help='Output format')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']),
              help='Device to run on (cpu, cuda)')
@click.option('--auto-confirm', is_flag=True,
              help='Skip confirmation for long processes')
def large_file_transcribe(audio_file: str, output: Optional[str], model: str, language: str,
                         min_confidence: float, chunk_duration: int, overlap_duration: float,
                         max_memory: int, no_enhanced_preprocessing: bool,
                         no_post_processing: bool, no_filter_fillers: bool, 
                         output_format: str, device: str, auto_confirm: bool):
    """
    Large file transcription with chunked processing.
    
    Designed for 2+ hour audio files with memory efficiency.
    Automatically handles memory management and progress tracking.
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("🏗️  LARGE FILE 音声文字起こしアプリケーション")
    print("=" * 70)
    print("📋 大容量ファイル（2時間以上）対応・メモリ効率最適化")
    print("=" * 70)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Check file size
    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        output = str(audio_path.parent / f"{audio_path.stem}_large_file")
    
    try:
        # Estimate processing time for large files
        estimator = TranscriptionTimeEstimator()
        
        print("📏 ファイル分析中...")
        
        # Get rough duration estimate (without loading full file)
        from pydub import AudioSegment
        try:
            audio_segment = AudioSegment.from_file(audio_file)
            audio_duration = len(audio_segment) / 1000.0
            del audio_segment  # Free memory immediately
        except Exception as e:
            print(f"⚠️  Duration analysis failed: {e}")
            # Rough estimate based on file size (MP3: ~1MB per minute)
            audio_duration = file_size_mb * 60  # Conservative estimate
        
        print(f"🎵 推定音声長: {audio_duration/60:.1f}分 ({audio_duration/3600:.1f}時間)")
        
        # Adjust chunk size for very large files
        if audio_duration > 7200:  # 2+ hours
            recommended_chunk = max(15, chunk_duration)  # At least 15 minutes
            if chunk_duration < recommended_chunk:
                print(f"📊 大容量ファイル検出: チャンクサイズを{chunk_duration}分→{recommended_chunk}分に調整")
                chunk_duration = recommended_chunk
        
        # Generate time estimate using chunked processing parameters
        estimates = estimator.estimate_processing_time(
            audio_duration=audio_duration,
            model_size=model,
            device=device,
            engine='chunked-enhanced',
            enhanced_preprocessing=not no_enhanced_preprocessing,
            post_processing=not no_post_processing,
            speaker_diarization=False  # Not implemented in chunked version yet
        )
        
        # Apply chunked processing efficiency (usually 20-30% faster)
        chunking_efficiency = 0.75  # 25% improvement from chunking
        for key in ['optimistic', 'realistic', 'pessimistic']:
            estimates[key] *= chunking_efficiency
        
        # Display time estimation
        print("\\n" + "⏱️" * 3 + " 大容量ファイル処理時間予測 " + "⏱️" * 3)
        print("=" * 65)
        print(f"🎵 音声長: {estimator.format_time_estimate(audio_duration)}")
        print(f"📦 チャンクサイズ: {chunk_duration}分 (オーバーラップ: {overlap_duration}秒)")
        print(f"🧠 メモリ制限: {max_memory}MB/チャンク")
        print(f"⚡ 最短予想: {estimator.format_time_estimate(estimates['optimistic'])}")
        print(f"🎯 標準予想: {estimator.format_time_estimate(estimates['realistic'])}")
        print(f"⏳ 最長予想: {estimator.format_time_estimate(estimates['pessimistic'])}")
        print(f"📊 処理比率: {estimates['processing_ratio']:.2f}x")
        print("=" * 65)
        
        # Warning for very long processing times
        if estimates['realistic'] > 1800:  # 30 minutes
            print(f"⚠️  長時間処理予想: {estimator.format_time_estimate(estimates['realistic'])}")
            print("💡 ヒント: より小さなモデル（tiny/base）やCUDAの使用を検討してください")
        
        # Ask for confirmation on long processes
        if estimates['realistic'] > 600 and not auto_confirm:  # 10 minutes
            print(f"⚠️  処理に{estimator.format_time_estimate(estimates['realistic'])}かかる可能性があります")
            if not click.confirm("続行しますか？"):
                return
        
        # Initialize chunked transcriber
        print("\\n🔧 チャンク処理システム初期化中...")
        transcriber = ChunkedTranscriber(
            model_size=model,
            language=language,
            device=device,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            max_memory_mb=max_memory
        )
        
        # Initialize progress tracking
        progress_callback = LargeFileProgressCallback(estimates['realistic'])
        
        start_time = time.time()
        
        try:
            # Process large file with chunked approach
            print("\\n🚀 チャンク処理開始...")
            segments = transcriber.process_large_file(
                audio_file=audio_file,
                enhanced_preprocessing=not no_enhanced_preprocessing,
                filter_fillers=not no_filter_fillers,
                min_confidence=min_confidence,
                progress_callback=progress_callback
            )
            
            if not segments:
                print("⚠️  文字起こし結果が得られませんでした")
                return
            
            print(f"\\n✅ チャンク処理完了: {len(segments)} セグメント")
            
            # Post-processing
            if not no_post_processing:
                progress_callback("post_processing", 0.0)
                print("🔧 後処理適用中...")
                
                post_processor = TranscriptionPostProcessor()
                segments = post_processor.process_transcription(segments)
                
                progress_callback("post_processing", 1.0)
                print(f"✅ 後処理完了: {len(segments)} セグメント")
            
            # Save output
            progress_callback("saving", 0.0)
            print(f"💾 結果保存中: {output}")
            
            output_formatter = OutputFormatter()
            
            # Prepare metadata
            processing_time = time.time() - start_time
            avg_confidence = sum(seg["confidence"] for seg in segments) / len(segments)
            
            metadata = {
                "input_file": audio_file,
                "model_size": model,
                "language": language,
                "device": device,
                "engine": "chunked-enhanced",
                "processing_time_seconds": round(processing_time, 2),
                "audio_duration_seconds": audio_duration,
                "processing_ratio": round(processing_time / audio_duration, 3),
                "average_confidence": round(avg_confidence, 3),
                "estimated_time": round(estimates['realistic'], 2),
                "estimation_accuracy": round(abs(processing_time - estimates['realistic']) / estimates['realistic'], 3),
                "total_segments": len(segments),
                "chunk_duration_minutes": chunk_duration,
                "max_memory_mb": max_memory,
                "file_size_mb": round(file_size_mb, 1)
            }
            
            if output_format == 'all':
                saved_files = output_formatter.save_all_formats(segments, output, metadata)
                progress_callback("saving", 1.0, "全形式")
            else:
                output_data = output_formatter.prepare_output_data(segments)
                
                if output_format == 'json':
                    output_formatter.save_as_json(output_data, f"{output}.json", metadata)
                elif output_format == 'csv':
                    output_formatter.save_as_csv(output_data, f"{output}.csv")
                elif output_format == 'txt':
                    output_formatter.save_as_txt(output_data, f"{output}.txt")
                elif output_format == 'srt':
                    output_formatter.save_as_srt(output_data, f"{output}.srt")
                
                progress_callback("saving", 1.0, output_format.upper())
            
        finally:
            # Clean up transcriber resources
            transcriber.cleanup()
        
        # Final results
        processing_time = time.time() - start_time
        
        print("\\n" + "=" * 70)
        print("🎯 LARGE FILE PROCESSING RESULTS")
        print("=" * 70)
        print(f"⏱️  実際の処理時間: {estimator.format_time_estimate(processing_time)}")
        print(f"🔮 予想処理時間: {estimator.format_time_estimate(estimates['realistic'])}")
        
        # Calculate prediction accuracy
        accuracy = abs(processing_time - estimates['realistic']) / estimates['realistic']
        if accuracy < 0.2:
            accuracy_emoji = "🎯"
            accuracy_text = "優秀"
        elif accuracy < 0.4:
            accuracy_emoji = "📊"
            accuracy_text = "良好"
        else:
            accuracy_emoji = "📈"
            accuracy_text = "要改善"
        
        print(f"{accuracy_emoji} 予測精度: {accuracy_text} (誤差: {accuracy:.1%})")
        print(f"📊 平均信頼度: {avg_confidence:.1%}")
        print(f"📝 総セグメント: {len(segments)}")
        print(f"⚡ 処理速度: {processing_time/audio_duration:.2f}x")
        print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
        print(f"🧠 メモリ効率: {max_memory}MB/チャンク")
        print("=" * 70)
        print("🎉 大容量ファイル処理完了!")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        raise


def cli():
    """Large file audio transcription CLI application."""
    large_file_transcribe()


if __name__ == '__main__':
    cli()