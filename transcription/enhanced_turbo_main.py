"""
Enhanced Turbo transcription system with improved speaker identification
and optimized data structures.
"""

import click
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from .enhanced_audio_processor import EnhancedAudioProcessor
from .post_processor import TranscriptionPostProcessor
from .enhanced_speaker_diarization import EnhancedSpeakerDiarizer, get_speaker_statistics
from .optimized_output_formatter import OptimizedOutputFormatter
from .precision_enhancer import AdvancedVAD
from .chunked_transcriber import ChunkedTranscriber
from .time_estimator import TranscriptionTimeEstimator


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='large-v3-turbo', 
              type=click.Choice(['large-v3-turbo', 'turbo', 'large', 'medium', 'base', 'small', 'tiny']),
              help='Whisper model size (default: large-v3-turbo)')
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
@click.option('--speaker-method', default='auto',
              type=click.Choice(['auto', 'pyannote', 'acoustic', 'clustering', 'off']),
              help='Speaker diarization method')
@click.option('--num-speakers', type=int,
              help='Expected number of speakers (optional)')
@click.option('--hf-token', help='Hugging Face token for pyannote models')
@click.option('--output-format', default='optimized',
              type=click.Choice(['legacy', 'compact', 'standard', 'extended', 'api', 'all']),
              help='Output data format')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']),
              help='Device to run on (cpu, cuda)')
@click.option('--auto-confirm', is_flag=True,
              help='Skip confirmation for long processes')
def enhanced_turbo_transcribe(
    audio_file: str, output: Optional[str], model: str, turbo_mode: bool,
    language: str, min_confidence: float, chunk_size: int, use_advanced_vad: bool,
    realtime_mode: bool, no_enhanced_preprocessing: bool, no_post_processing: bool,
    speaker_method: str, num_speakers: Optional[int], hf_token: Optional[str],
    output_format: str, device: str, auto_confirm: bool
):
    """
    Enhanced Turbo transcription with improved speaker identification.
    
    Features:
    - Multiple speaker diarization methods with fallbacks
    - Optimized data structures with reduced redundancy
    - Purpose-specific output formats
    - Enhanced quality metrics and validation
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("⚡ ENHANCED TURBO 音声文字起こしシステム")
    print("=" * 70)
    if turbo_mode:
        print("🚀 Turbo最適化モード: 高速・高精度バランス")
    if realtime_mode:
        print("🔴 リアルタイムモード: 低遅延処理")
    if speaker_method != 'off':
        print(f"👥 話者識別方式: {speaker_method}")
    print(f"📊 出力形式: {output_format}")
    print("=" * 70)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        suffix = "_enhanced_turbo"
        if realtime_mode:
            suffix += "_realtime"
        if speaker_method != 'off':
            suffix += f"_{speaker_method}"
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
            if chunk_size < 10:
                chunk_size = 10
                print(f"📊 Turbo最適化: チャンクサイズを{chunk_size}秒に調整")
            elif chunk_size > 15:
                chunk_size = 15
                print(f"📊 Turbo最適化: チャンクサイズを{chunk_size}秒に調整")
        
        # Time estimation with Turbo optimizations
        estimator = TranscriptionTimeEstimator()
        engine_type = "enhanced-turbo" if turbo_mode else "enhanced"
        
        estimates = estimator.estimate_processing_time(
            audio_duration=audio_duration,
            model_size=model,
            device=device,
            engine=engine_type,
            enhanced_preprocessing=not no_enhanced_preprocessing,
            post_processing=not no_post_processing,
            speaker_diarization=(speaker_method != 'off')
        )
        
        # Apply Turbo speed boost
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
            print(f"🏗️  大容量ファイル: Enhanced Turbo+チャンク処理")
            final_segments = process_large_file_enhanced_turbo(
                audio_file=audio_file,
                model=model,
                language=language,
                device=device,
                chunk_duration_seconds=chunk_size,
                turbo_mode=turbo_mode,
                use_advanced_vad=use_advanced_vad,
                enhanced_preprocessing=not no_enhanced_preprocessing,
                min_confidence=min_confidence,
                speaker_method=speaker_method,
                num_speakers=num_speakers,
                hf_token=hf_token
            )
        else:
            print(f"⚡ 標準ファイル: Enhanced Turbo処理")
            final_segments = process_standard_file_enhanced_turbo(
                audio_file=audio_file,
                model=model,
                language=language,
                device=device,
                chunk_size=chunk_size,
                turbo_mode=turbo_mode,
                use_advanced_vad=use_advanced_vad,
                enhanced_preprocessing=not no_enhanced_preprocessing,
                post_processing=not no_post_processing,
                min_confidence=min_confidence,
                speaker_method=speaker_method,
                num_speakers=num_speakers,
                hf_token=hf_token,
                realtime_mode=realtime_mode
            )
        
        if not final_segments:
            print("⚠️  文字起こし結果が得られませんでした")
            return
        
        print(f"\\n✅ Enhanced Turbo処理完了: {len(final_segments)} セグメント")
        
        # Calculate metrics
        processing_time = time.time() - start_time
        avg_confidence = sum(seg["confidence"] for seg in final_segments) / len(final_segments)
        total_text_length = sum(len(seg["text"]) for seg in final_segments)
        actual_speedup = 1.0 / (processing_time / audio_duration) if audio_duration > 0 else 0
        
        # Enhanced metadata
        metadata = {
            "input_file": audio_file,
            "model_size": model,
            "language": language,
            "device": device,
            "engine": "enhanced-turbo",
            "turbo_mode": turbo_mode,
            "realtime_mode": realtime_mode,
            "chunk_size_seconds": chunk_size,
            "speaker_method": speaker_method,
            "expected_speakers": num_speakers,
            "techniques_applied": {
                "turbo_optimization": turbo_mode,
                "advanced_vad": use_advanced_vad,
                "enhanced_preprocessing": not no_enhanced_preprocessing,
                "post_processing": not no_post_processing,
                "enhanced_speaker_diarization": speaker_method != 'off',
                "chunked_processing": use_chunked_processing,
                "realtime_optimization": realtime_mode
            },
            "quality_metrics": {
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
        }
        
        # Get speaker statistics
        speaker_stats = get_speaker_statistics(final_segments)
        metadata["speaker_statistics"] = speaker_stats
        
        # Save Enhanced output
        print(f"\\n💾 Enhanced結果保存中: {output}")
        output_formatter = OptimizedOutputFormatter()
        
        if output_format == 'all':
            # Save all format variants
            saved_files = {}
            for variant in ['compact', 'standard', 'extended', 'api']:
                file_path = output_formatter.save_optimized_format(
                    final_segments, output, variant, metadata
                )
                saved_files[variant] = file_path
                
                # Also save CSV for standard and extended
                if variant in ['standard', 'extended']:
                    csv_path = output_formatter.save_optimized_csv(
                        final_segments, f"{output}_{variant}", variant
                    )
                    saved_files[f"{variant}_csv"] = csv_path
            
            print("📁 保存ファイル:")
            for format_name, file_path in saved_files.items():
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   - {format_name.upper()}: {file_path} ({file_size:.1f}KB)")
        
        elif output_format == 'legacy':
            # Use original output formatter for backward compatibility
            from .output_formatter import OutputFormatter
            legacy_formatter = OutputFormatter()
            saved_files = legacy_formatter.save_all_formats(final_segments, output, metadata)
            print("📁 保存ファイル (Legacy形式):")
            for format_name, file_path in saved_files.items():
                print(f"   - {format_name.upper()}: {file_path}")
        
        else:
            # Save specific format
            file_path = output_formatter.save_optimized_format(
                final_segments, output, output_format, metadata
            )
            csv_path = output_formatter.save_optimized_csv(
                final_segments, output, output_format
            )
            
            file_size = os.path.getsize(file_path) / 1024
            csv_size = os.path.getsize(csv_path) / 1024
            print(f"📁 保存ファイル:")
            print(f"   - JSON: {file_path} ({file_size:.1f}KB)")
            print(f"   - CSV: {csv_path} ({csv_size:.1f}KB)")
        
        # Display Enhanced results
        print("\\n" + "=" * 70)
        print("⚡ ENHANCED TURBO RESULTS")
        print("=" * 70)
        print(f"📊 平均信頼度: {avg_confidence:.3f} ({avg_confidence:.1%})")
        print(f"📝 総セグメント: {len(final_segments)}")
        print(f"📄 総文字数: {total_text_length}")
        
        # Enhanced speaker information
        known_speakers = [s for s in speaker_stats.keys() if s != "SPEAKER_UNKNOWN"]
        if known_speakers:
            print(f"👥 検出話者数: {len(known_speakers)}")
            for speaker_id in sorted(known_speakers):
                stats = speaker_stats[speaker_id]
                print(f"   - {speaker_id}: {stats['segment_count']}セグメント, "
                      f"{stats['total_duration']:.1f}秒 ({stats['avg_confidence']:.1%}信頼度)")
        else:
            print(f"👥 話者識別: 実行されませんでした")
        
        print(f"⏱️  処理時間: {processing_time:.1f}秒 ({processing_time/60:.1f}分)")
        print(f"🎧 音声長: {audio_duration/60:.1f}分")
        print(f"⚡ 実測速度: {actual_speedup:.1f}x")
        print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
        
        # Show optimization effects
        if turbo_mode:
            print(f"\\n🚀 Turbo最適化効果:")
            print(f"   ✅ 高速処理アルゴリズム")
            print(f"   ✅ {chunk_size}秒最適チャンク")
            if use_advanced_vad:
                print(f"   ✅ 高度VAD処理")
        
        if speaker_method != 'off':
            print(f"\\n👥 話者識別: {speaker_method} 方式")
            if known_speakers:
                print(f"   ✅ {len(known_speakers)} 話者を成功検出")
            else:
                print(f"   ⚠️  話者識別に失敗、単一話者として処理")
        
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
        
        print(f"\\n🎯 品質評価: {quality}")
        
        # Data format information
        if output_format != 'legacy':
            format_info = output_formatter.get_format_comparison()[output_format]
            print(f"\\n📊 データ形式: {output_format}")
            print(f"   - {format_info['description']}")
            print(f"   - 最適用途: {format_info['best_for']}")
        
        print("=" * 70)
        print(f"🎉 Enhanced Turbo転写完了! 平均信頼度: {avg_confidence:.1%}")
        
        if turbo_mode and speaker_method != 'off':
            print(f"⚡ Turbo+Enhanced話者識別により高速・高精度を両立!")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        raise

def process_standard_file_enhanced_turbo(
    audio_file: str, model: str, language: str, device: str, chunk_size: int,
    turbo_mode: bool, use_advanced_vad: bool, enhanced_preprocessing: bool,
    post_processing: bool, min_confidence: float, speaker_method: str,
    num_speakers: Optional[int], hf_token: Optional[str], realtime_mode: bool
) -> List[Dict]:
    """Process standard file with enhanced turbo optimizations."""
    
    # Enhanced audio processing
    audio_processor = EnhancedAudioProcessor()
    
    if realtime_mode or not enhanced_preprocessing:
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
        print(f"🎧 Enhanced Turbo音声処理")
        audio_data, sample_rate = audio_processor.advanced_preprocess_audio(
            audio_file,
            enable_noise_reduction=True,
            enable_speech_enhancement=True,
            enable_spectral_norm=not turbo_mode,
            enable_volume_adjustment=True,
            enable_silence_removal=False,
            memory_efficient=turbo_mode
        )
    
    # Enhanced VAD
    if use_advanced_vad:
        print("🎙️  Enhanced VAD適用中...")
        vad = AdvancedVAD(threshold=0.5, min_speech_duration=150)
        audio_data = vad.apply_advanced_vad(audio_data, sample_rate)
    
    # Enhanced transcription
    from .faster_transcriber import FasterTranscriber
    transcriber = FasterTranscriber(
        model_size=model,
        language=language,
        device=device
    )
    
    print(f"⚡ Enhanced Turbo転写実行中 (model: {model})...")
    segments = transcriber.process_transcription(
        audio_data,
        sample_rate,
        filter_confidence=True,
        filter_fillers=True,
        min_confidence=min_confidence
    )
    
    # Convert to dict format
    transcription_segments = [
        {
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "start_seconds": seg.start_time,
            "end_seconds": seg.end_time,
            "text": seg.text,
            "confidence": seg.confidence
        }
        for seg in segments
    ]
    
    # Enhanced post-processing
    if post_processing:
        print("🔧 Enhanced後処理適用中...")
        post_processor = TranscriptionPostProcessor()
        transcription_segments = post_processor.process_transcription(transcription_segments)
        print(f"✅ 高速後処理完了: {len(transcription_segments)} セグメント")
    
    # Enhanced speaker diarization
    if speaker_method != 'off':
        print(f"👥 Enhanced話者識別実行中: {speaker_method} 方式")
        speaker_diarizer = EnhancedSpeakerDiarizer(
            use_auth_token=hf_token,
            method=speaker_method
        )
        
        if speaker_diarizer.is_available():
            speaker_segments = speaker_diarizer.diarize_audio(
                audio_data, sample_rate, num_speakers
            )
            transcription_segments = speaker_diarizer.assign_speakers_to_transcription(
                transcription_segments, speaker_segments
            )
        else:
            print("⚠️  話者識別が利用できません。SPEAKER_UNKNOWN を割り当てます。")
            for seg in transcription_segments:
                seg["speaker_id"] = "SPEAKER_UNKNOWN"
                seg["speaker_confidence"] = 0.0
    else:
        # No speaker diarization
        for seg in transcription_segments:
            seg["speaker_id"] = "SPEAKER_UNKNOWN"
            seg["speaker_confidence"] = None
    
    return transcription_segments

def process_large_file_enhanced_turbo(
    audio_file: str, model: str, language: str, device: str,
    chunk_duration_seconds: int, turbo_mode: bool, use_advanced_vad: bool,
    enhanced_preprocessing: bool, min_confidence: float,
    speaker_method: str, num_speakers: Optional[int], hf_token: Optional[str]
) -> List[Dict]:
    """Process large files with enhanced turbo optimizations."""
    print("🏗️  大容量ファイル Enhanced Turbo処理開始...")
    
    chunk_duration_minutes = max(1, chunk_duration_seconds // 60)
    
    # Initialize Enhanced Turbo-optimized chunked transcriber
    chunked_transcriber = ChunkedTranscriber(
        model_size=model,
        language=language,
        device=device,
        chunk_duration=chunk_duration_minutes,
        overlap_duration=2.0,
        max_memory_mb=512 if turbo_mode else 1024
    )
    
    try:
        # Process with Enhanced Turbo settings
        segments = chunked_transcriber.process_large_file(
            audio_file=audio_file,
            enhanced_preprocessing=enhanced_preprocessing,
            filter_fillers=True,
            min_confidence=min_confidence
        )
        
        # Convert to proper format and add speaker information
        final_segments = []
        for seg in segments:
            final_segments.append({
                "start_time": seg.get("start_time", 0),
                "end_time": seg.get("end_time", 0),
                "start_seconds": seg.get("start_time", 0),
                "end_seconds": seg.get("end_time", 0),
                "text": seg.get("text", ""),
                "confidence": seg.get("confidence", 0),
                "speaker_id": "SPEAKER_UNKNOWN",  # Large file processing doesn't include speaker ID
                "speaker_confidence": None
            })
        
        print(f"✅ Enhanced Turbo大容量処理完了: {len(final_segments)} セグメント")
        return final_segments
        
    finally:
        chunked_transcriber.cleanup()

def cli():
    """Enhanced Turbo audio transcription CLI application."""
    enhanced_turbo_transcribe()

if __name__ == '__main__':
    cli()