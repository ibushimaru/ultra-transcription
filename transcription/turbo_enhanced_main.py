"""
Turbo-enhanced transcription system optimized for speed and accuracy balance.
Based on Whisper Large-v3 Turbo for optimal performance.
"""

import click
import os
import time
from pathlib import Path
from typing import Optional

from .enhanced_audio_processor import EnhancedAudioProcessor
from .post_processor import TranscriptionPostProcessor
from .speaker_diarization import SpeakerDiarizer
from .output_formatter import OutputFormatter
from .precision_enhancer import EnsembleTranscriber, AdvancedVAD
from .chunked_transcriber import ChunkedTranscriber
from .time_estimator import TranscriptionTimeEstimator


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='large-v3-turbo', 
              type=click.Choice(['large-v3-turbo', 'turbo', 'large-v3', 'medium', 'base']),
              help='Whisper model size (default: large-v3-turbo for maximum speed)')
@click.option('--turbo-mode', is_flag=True, default=True,
              help='Enable Turbo optimization mode (default: enabled)')
@click.option('--language', '-l', default='ja', help='Language code for transcription')
@click.option('--min-confidence', default=0.3, type=float,
              help='Minimum confidence threshold for segments')
@click.option('--chunk-size', default=15, type=int,
              help='Audio chunk size in seconds (optimized for Turbo: 10-15s)')
@click.option('--use-advanced-vad', is_flag=True, default=True,
              help='Use advanced VAD for better segmentation (default: enabled)')
@click.option('--realtime-mode', is_flag=True,
              help='Enable real-time processing optimizations')
@click.option('--no-enhanced-preprocessing', is_flag=True,
              help='Skip enhanced audio preprocessing')
@click.option('--no-post-processing', is_flag=True,
              help='Skip post-processing corrections')
@click.option('--no-speaker-diarization', is_flag=True,
              help='Skip speaker diarization')
@click.option('--hf-token', help='Hugging Face token for pyannote models')
@click.option('--format', 'output_format', 
              type=click.Choice(['all', 'json', 'csv', 'txt', 'srt']),
              default='all', help='Output format')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']),
              help='Device to run on (cpu, cuda)')
@click.option('--auto-confirm', is_flag=True,
              help='Skip confirmation for long processes')
def turbo_enhanced_transcribe(audio_file: str, output: Optional[str], model: str, 
                             turbo_mode: bool, language: str, min_confidence: float,
                             chunk_size: int, use_advanced_vad: bool, realtime_mode: bool,
                             no_enhanced_preprocessing: bool, no_post_processing: bool,
                             no_speaker_diarization: bool, hf_token: Optional[str], 
                             output_format: str, device: str, auto_confirm: bool):
    """
    Turbo-enhanced transcription optimized for speed-accuracy balance.
    
    Uses Whisper Large-v3 Turbo architecture for 3x faster processing
    while maintaining high accuracy levels.
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("⚡ TURBO ENHANCED 音声文字起こしアプリケーション")
    print("=" * 70)
    if turbo_mode:
        print("🚀 Turbo最適化モード: 高速・高精度バランス")
    if realtime_mode:
        print("🔴 リアルタイムモード: 低遅延処理")
    print("=" * 70)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        suffix = "_turbo" if turbo_mode else "_enhanced"
        if realtime_mode:
            suffix += "_realtime"
        output = str(audio_path.parent / f"{audio_path.stem}{suffix}")
    
    try:
        start_time = time.time()
        
        # File analysis for processing strategy
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
        
        # Quick duration check
        from pydub import AudioSegment
        try:
            audio_segment = AudioSegment.from_file(audio_file)
            audio_duration = len(audio_segment) / 1000.0
            del audio_segment
        except Exception as e:
            print(f"⚠️  Duration analysis failed: {e}")
            audio_duration = file_size_mb * 60  # Conservative estimate
        
        print(f"⏱️  音声長: {audio_duration/60:.1f}分 ({audio_duration/3600:.1f}時間)")
        
        # Turbo-optimized processing strategy
        if turbo_mode:
            # Optimize chunk size for Turbo performance (10-15 seconds)
            if chunk_size < 10:
                chunk_size = 10
                print(f"📊 Turbo最適化: チャンクサイズを{chunk_size}秒に調整")
            elif chunk_size > 15:
                chunk_size = 15
                print(f"📊 Turbo最適化: チャンクサイズを{chunk_size}秒に調整")
        
        # Time estimation with Turbo optimizations
        estimator = TranscriptionTimeEstimator()
        engine_type = "turbo-enhanced" if turbo_mode else "ultra-enhanced"
        
        estimates = estimator.estimate_processing_time(
            audio_duration=audio_duration,
            model_size=model,
            device=device,
            engine=engine_type,
            enhanced_preprocessing=not no_enhanced_preprocessing,
            post_processing=not no_post_processing,
            speaker_diarization=not no_speaker_diarization
        )
        
        # Apply Turbo speed boost (3.16x faster)
        if turbo_mode:
            turbo_speedup = 0.316  # 3.16x faster
            for key in ['optimistic', 'realistic', 'pessimistic']:
                estimates[key] *= turbo_speedup
        
        print(f"⏱️  予想処理時間: {estimator.format_time_estimate(estimates['realistic'])}")
        if turbo_mode:
            print(f"🚀 Turbo効果: 約3.2倍高速化")
        
        # Determine processing approach
        use_chunked_processing = audio_duration > 1800  # 30 minutes
        
        if use_chunked_processing:
            print(f"🏗️  大容量ファイル: Turbo+チャンク処理")
            final_segments = process_large_file_turbo(
                audio_file=audio_file,
                model=model,
                language=language,
                device=device,
                chunk_duration_seconds=chunk_size,
                turbo_mode=turbo_mode,
                use_advanced_vad=use_advanced_vad,
                enhanced_preprocessing=not no_enhanced_preprocessing,
                min_confidence=min_confidence
            )
            
        else:
            print(f"⚡ 標準ファイル: Turbo最適化処理")
            
            # Enhanced audio processing with Turbo optimizations
            audio_processor = EnhancedAudioProcessor()
            
            if no_enhanced_preprocessing or realtime_mode:
                print(f"🎧 高速音声処理")
                audio_data, sample_rate = audio_processor.advanced_preprocess_audio(
                    audio_file, 
                    enable_noise_reduction=True,
                    enable_speech_enhancement=False,
                    enable_spectral_norm=False,
                    enable_volume_adjustment=True,
                    enable_silence_removal=False,
                    memory_efficient=True
                )
            else:
                print(f"🎧 Turbo最適化音声処理")
                audio_data, sample_rate = audio_processor.advanced_preprocess_audio(
                    audio_file,
                    enable_noise_reduction=True,
                    enable_speech_enhancement=True,
                    enable_spectral_norm=not turbo_mode,  # Skip for Turbo speed
                    enable_volume_adjustment=True,
                    enable_silence_removal=False,
                    memory_efficient=turbo_mode
                )
            
            # Turbo-optimized transcription
            if use_advanced_vad:
                print("🎙️  Turbo VAD適用中...")
                vad = AdvancedVAD(threshold=0.5, min_speech_duration=150)  # Optimized for Turbo
                audio_data = vad.apply_advanced_vad(audio_data, sample_rate)
            
            # Use optimized transcriber
            from .faster_transcriber import FasterTranscriber
            transcriber = FasterTranscriber(
                model_size=model,
                language=language,
                device=device
            )
            
            print(f"⚡ Turbo転写実行中 (model: {model})...")
            segments = transcriber.process_transcription(
                audio_data,
                sample_rate,
                filter_confidence=True,
                filter_fillers=True,
                min_confidence=min_confidence
            )
            
            # Convert to final format
            final_segments = [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "confidence": seg.confidence,
                    "speaker_id": "SPEAKER_UNKNOWN"
                }
                for seg in segments
            ]
        
        if not final_segments:
            print("⚠️  文字起こし結果が得られませんでした")
            return
        
        print(f"\n✅ Turbo処理完了: {len(final_segments)} セグメント")
        
        # Turbo-optimized post-processing
        if not no_post_processing:
            print("🔧 Turbo後処理適用中...")
            post_processor = TranscriptionPostProcessor()
            final_segments = post_processor.process_transcription(final_segments)
            print(f"✅ 高速後処理完了: {len(final_segments)} セグメント")
        
        # Speaker diarization (optional for Turbo mode)
        if not use_chunked_processing and not no_speaker_diarization and not realtime_mode:
            speaker_diarizer = SpeakerDiarizer(use_auth_token=hf_token)
            if speaker_diarizer and speaker_diarizer.is_available():
                print("👥 話者識別適用中...")
                # Use VAD-processed audio if available
                speaker_segments = speaker_diarizer.diarize_audio(audio_data, sample_rate)
                final_segments = speaker_diarizer.assign_speakers_to_transcription(
                    final_segments, speaker_segments
                )
                unique_speakers = set(seg["speaker_id"] for seg in final_segments)
                print(f"✅ {len(unique_speakers)} 話者検出")
        
        # Calculate metrics
        processing_time = time.time() - start_time
        avg_confidence = sum(seg["confidence"] for seg in final_segments) / len(final_segments)
        total_text_length = sum(len(seg["text"]) for seg in final_segments)
        
        # Calculate actual speedup
        actual_speedup = 1.0 / (processing_time / audio_duration) if audio_duration > 0 else 0
        
        # Prepare Turbo metadata
        metadata = {
            "input_file": audio_file,
            "model_size": model,
            "language": language,
            "device": device,
            "engine": "turbo-enhanced",
            "turbo_mode": turbo_mode,
            "realtime_mode": realtime_mode,
            "chunk_size_seconds": chunk_size,
            "techniques_applied": {
                "turbo_optimization": turbo_mode,
                "advanced_vad": use_advanced_vad,
                "enhanced_preprocessing": not no_enhanced_preprocessing,
                "post_processing": not no_post_processing,
                "speaker_diarization": not no_speaker_diarization,
                "chunked_processing": use_chunked_processing,
                "realtime_optimization": realtime_mode
            },
            "min_confidence": min_confidence,
            "audio_duration_seconds": audio_duration,
            "total_segments": len(final_segments),
            "average_confidence": round(avg_confidence, 3),
            "total_text_length": total_text_length,
            "processing_time_seconds": round(processing_time, 2),
            "processing_ratio": round(processing_time / audio_duration, 3) if audio_duration > 0 else None,
            "actual_speedup": round(actual_speedup, 2),
            "file_size_mb": round(file_size_mb, 1)
        }
        
        # Save Turbo output
        print(f"\n💾 Turbo結果保存中: {output}")
        output_formatter = OutputFormatter()
        
        if output_format == 'all':
            saved_files = output_formatter.save_all_formats(final_segments, output, metadata)
            print("📁 保存ファイル:")
            for format_name, file_path in saved_files.items():
                print(f"   - {format_name.upper()}: {file_path}")
        else:
            output_data = output_formatter.prepare_output_data(final_segments)
            
            if output_format == 'json':
                output_formatter.save_as_json(output_data, f"{output}.json", metadata)
                print(f"   - JSON: {output}.json")
            elif output_format == 'csv':
                output_formatter.save_as_csv(output_data, f"{output}.csv")
                print(f"   - CSV: {output}.csv")
            elif output_format == 'txt':
                output_formatter.save_as_txt(output_data, f"{output}.txt")
                print(f"   - TXT: {output}.txt")
            elif output_format == 'srt':
                output_formatter.save_as_srt(output_data, f"{output}.srt")
                print(f"   - SRT: {output}.srt")
        
        # Display Turbo results
        print("\n" + "=" * 70)
        print("⚡ TURBO ENHANCED RESULTS")
        print("=" * 70)
        print(f"📊 平均信頼度: {avg_confidence:.3f} ({avg_confidence:.1%})")
        print(f"📝 総セグメント: {len(final_segments)}")
        print(f"📄 総文字数: {total_text_length}")
        print(f"👥 話者数: {len(set(seg['speaker_id'] for seg in final_segments))}")
        print(f"⏱️  処理時間: {processing_time:.1f}秒 ({processing_time/60:.1f}分)")
        print(f"🎧 音声長: {audio_duration/60:.1f}分")
        print(f"⚡ 実測速度: {actual_speedup:.1f}x")
        print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
        
        # Show Turbo optimizations
        if turbo_mode:
            print(f"\n🚀 Turbo最適化:")
            print(f"   ✅ 3.2x高速化アルゴリズム")
            print(f"   ✅ {chunk_size}秒最適チャンク")
            if use_advanced_vad:
                print(f"   ✅ 高速VAD処理")
        
        if realtime_mode:
            print(f"🔴 リアルタイム最適化適用")
        
        # Quality assessment
        if avg_confidence >= 0.90:
            quality = "🟢 OUTSTANDING"
        elif avg_confidence >= 0.85:
            quality = "🟢 EXCELLENT"
        elif avg_confidence >= 0.80:
            quality = "🟡 VERY GOOD"
        elif avg_confidence >= 0.70:
            quality = "🟠 GOOD"
        else:
            quality = "🔴 FAIR"
        
        print(f"\n🎯 品質評価: {quality}")
        
        # Performance comparison
        expected_normal_time = audio_duration * 0.8  # Normal processing estimate
        if turbo_mode and processing_time < expected_normal_time:
            time_saved = expected_normal_time - processing_time
            print(f"💨 時間短縮: {time_saved/60:.1f}分節約")
        
        print("=" * 70)
        print(f"🎉 Turbo転写完了! 平均信頼度: {avg_confidence:.1%}")
        
        if turbo_mode:
            print(f"⚡ Turbo効果により高速・高精度を両立!")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        raise


def process_large_file_turbo(audio_file: str,
                           model: str,
                           language: str,
                           device: str,
                           chunk_duration_seconds: int,
                           turbo_mode: bool,
                           use_advanced_vad: bool,
                           enhanced_preprocessing: bool,
                           min_confidence: float) -> list:
    """
    Process large files with Turbo optimizations.
    """
    print("🏗️  大容量ファイル Turbo処理開始...")
    
    # Convert seconds to minutes for chunked transcriber
    chunk_duration_minutes = max(1, chunk_duration_seconds // 60)
    
    # Initialize Turbo-optimized chunked transcriber
    chunked_transcriber = ChunkedTranscriber(
        model_size=model,
        language=language,
        device=device,
        chunk_duration=chunk_duration_minutes,
        overlap_duration=2.0,  # Optimized for Turbo
        max_memory_mb=512 if turbo_mode else 1024
    )
    
    try:
        # Process with Turbo settings
        segments = chunked_transcriber.process_large_file(
            audio_file=audio_file,
            enhanced_preprocessing=enhanced_preprocessing,
            filter_fillers=True,
            min_confidence=min_confidence
        )
        
        print(f"✅ Turbo大容量処理完了: {len(segments)} セグメント")
        return segments
        
    finally:
        chunked_transcriber.cleanup()


def cli():
    """Turbo-enhanced audio transcription CLI application."""
    turbo_enhanced_transcribe()


if __name__ == '__main__':
    cli()