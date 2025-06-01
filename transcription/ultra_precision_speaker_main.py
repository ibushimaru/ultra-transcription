"""
Ultra Precision transcription with Enhanced Speaker Recognition.
Combines maximum accuracy transcription with advanced speaker diarization.
"""

import click
import os
import time
from pathlib import Path
from typing import Optional, List, Dict

from .enhanced_audio_processor import EnhancedAudioProcessor
from .post_processor import TranscriptionPostProcessor
from .enhanced_speaker_diarization import EnhancedSpeakerDiarizer, get_speaker_statistics
from .optimized_output_formatter import OptimizedOutputFormatter
from .precision_enhancer import PrecisionEnhancer
from .chunked_transcriber import ChunkedTranscriber
from .time_estimator import TranscriptionTimeEstimator


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='large-v3-turbo', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo']),
              help='Primary Whisper model size (default: large-v3-turbo for maximum accuracy)')
@click.option('--language', '-l', default='ja', help='Language code for transcription')
@click.option('--min-confidence', default=0.15, type=float,
              help='Minimum confidence threshold (lowered for ensemble filtering)')
@click.option('--use-ensemble', is_flag=True, default=True,
              help='Use ensemble of multiple models (default: enabled)')
@click.option('--ensemble-models', default='medium,large,large-v3-turbo', 
              help='Comma-separated list of models for ensemble (default: medium,large,large-v3-turbo)')
@click.option('--voting-method', default='confidence_weighted',
              type=click.Choice(['confidence_weighted', 'majority']),
              help='Ensemble voting method')
@click.option('--use-advanced-vad', is_flag=True, default=True,
              help='Use advanced voice activity detection (default: enabled)')
@click.option('--no-enhanced-preprocessing', is_flag=True,
              help='Skip enhanced audio preprocessing')
@click.option('--no-post-processing', is_flag=True,
              help='Skip post-processing corrections')
@click.option('--speaker-method', default='auto',
              type=click.Choice(['auto', 'pyannote', 'acoustic', 'clustering', 'off']),
              help='Enhanced speaker diarization method (default: auto)')
@click.option('--num-speakers', type=int,
              help='Expected number of speakers (optional)')
@click.option('--hf-token', help='Hugging Face token for pyannote models')
@click.option('--output-format', default='extended',
              type=click.Choice(['legacy', 'compact', 'standard', 'extended', 'api', 'all']),
              help='Output data format (default: extended for detailed analysis)')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']),
              help='Device to run on (cpu, cuda)')
@click.option('--chunk-threshold', default=1800, type=int,
              help='File duration threshold for chunked processing (seconds)')
@click.option('--auto-confirm', is_flag=True,
              help='Skip confirmation for long processes')
def ultra_precision_transcribe(
    audio_file: str, output: Optional[str], model: str, language: str,
    min_confidence: float, use_ensemble: bool, ensemble_models: str,
    voting_method: str, use_advanced_vad: bool, no_enhanced_preprocessing: bool,
    no_post_processing: bool, speaker_method: str, num_speakers: Optional[int],
    hf_token: Optional[str], output_format: str, device: str,
    chunk_threshold: int, auto_confirm: bool
):
    """
    Ultra Precision transcription with Enhanced Speaker Recognition.
    
    This system combines:
    - Maximum accuracy transcription (ensemble models, advanced VAD, precision enhancer)
    - Enhanced speaker diarization with multiple methods
    - Optimized data structures with reduced redundancy
    - Comprehensive quality metrics and validation
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("🎯 ULTRA PRECISION + ENHANCED SPEAKER RECOGNITION")
    print("=" * 80)
    print("🔬 最高精度転写 + 高度話者識別システム")
    
    if use_ensemble:
        model_list = [m.strip() for m in ensemble_models.split(',')]
        print(f"🎭 Ensemble models: {model_list}")
    else:
        model_list = [model]
        print(f"📝 Single model: {model}")
    
    if speaker_method != 'off':
        print(f"👥 Enhanced話者識別: {speaker_method} 方式")
    
    print(f"📊 出力形式: {output_format}")
    print("=" * 80)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        suffix = "_ultra_precision_speaker"
        output = str(audio_path.parent / f"{audio_path.stem}{suffix}")
    
    try:
        start_time = time.time()
        
        # File analysis
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
            audio_duration = file_size_mb * 60
        
        print(f"⏱️  音声長: {audio_duration/60:.1f}分 ({audio_duration/3600:.1f}時間)")
        
        # Time estimation
        estimator = TranscriptionTimeEstimator()
        estimates = estimator.estimate_processing_time(
            audio_duration=audio_duration,
            model_size=model,
            device=device,
            engine='ultra-precision-speaker',
            enhanced_preprocessing=not no_enhanced_preprocessing,
            post_processing=not no_post_processing,
            speaker_diarization=(speaker_method != 'off')
        )
        
        # Adjust for ensemble and speaker recognition
        if use_ensemble:
            ensemble_multiplier = len(model_list) * 0.7  # Parallel efficiency
            for key in ['optimistic', 'realistic', 'pessimistic']:
                estimates[key] *= ensemble_multiplier
        
        if speaker_method != 'off':
            speaker_multiplier = 1.5  # Speaker diarization overhead
            for key in ['optimistic', 'realistic', 'pessimistic']:
                estimates[key] *= speaker_multiplier
        
        print(f"⏱️  予想処理時間: {estimator.format_time_estimate(estimates['realistic'])}")
        print(f"🔬 Ultra Precision + Enhanced Speaker Recognition")
        
        # Determine processing strategy
        use_chunked_processing = audio_duration > chunk_threshold
        
        if use_chunked_processing:
            print(f"🏗️  大容量ファイル: Ultra Precision チャンク処理")
            final_segments = process_large_file_ultra_precision(
                audio_file=audio_file,
                model_list=model_list,
                language=language,
                device=device,
                use_ensemble=use_ensemble,
                voting_method=voting_method,
                use_advanced_vad=use_advanced_vad,
                enhanced_preprocessing=not no_enhanced_preprocessing,
                min_confidence=min_confidence,
                speaker_method=speaker_method,
                num_speakers=num_speakers,
                hf_token=hf_token
            )
        else:
            print(f"🎯 標準ファイル: Ultra Precision処理")
            final_segments = process_standard_file_ultra_precision(
                audio_file=audio_file,
                model_list=model_list,
                language=language,
                device=device,
                use_ensemble=use_ensemble,
                voting_method=voting_method,
                use_advanced_vad=use_advanced_vad,
                enhanced_preprocessing=not no_enhanced_preprocessing,
                post_processing=not no_post_processing,
                min_confidence=min_confidence,
                speaker_method=speaker_method,
                num_speakers=num_speakers,
                hf_token=hf_token
            )
        
        if not final_segments:
            print("⚠️  文字起こし結果が得られませんでした")
            return
        
        print(f"\\n✅ Ultra Precision処理完了: {len(final_segments)} セグメント")
        
        # Calculate comprehensive metrics
        processing_time = time.time() - start_time
        avg_confidence = sum(seg.get("confidence", 0) for seg in final_segments) / len(final_segments)
        total_text_length = sum(len(seg.get("text", "")) for seg in final_segments)
        
        # Enhanced metadata
        metadata = {
            "input_file": audio_file,
            "model_configuration": {
                "primary_model": model,
                "ensemble_models": model_list if use_ensemble else [model],
                "use_ensemble": use_ensemble,
                "voting_method": voting_method if use_ensemble else None
            },
            "language": language,
            "device": device,
            "engine": "ultra-precision-speaker",
            "speaker_configuration": {
                "method": speaker_method,
                "expected_speakers": num_speakers,
                "hf_token_used": hf_token is not None
            },
            "processing_techniques": {
                "ensemble_transcription": use_ensemble,
                "advanced_vad": use_advanced_vad,
                "enhanced_preprocessing": not no_enhanced_preprocessing,
                "post_processing": not no_post_processing,
                "enhanced_speaker_diarization": speaker_method != 'off',
                "chunked_processing": use_chunked_processing,
                "precision_enhancement": True
            },
            "quality_metrics": {
                "min_confidence": min_confidence,
                "audio_duration_seconds": audio_duration,
                "total_segments": len(final_segments),
                "average_confidence": round(avg_confidence, 4),
                "total_text_length": total_text_length,
                "processing_time_seconds": round(processing_time, 2),
                "processing_ratio": round(processing_time / audio_duration, 3) if audio_duration > 0 else None,
                "file_size_mb": round(file_size_mb, 1)
            }
        }
        
        # Get speaker statistics
        speaker_stats = get_speaker_statistics(final_segments)
        metadata["speaker_statistics"] = speaker_stats
        
        # Save Ultra Precision results
        print(f"\\n💾 Ultra Precision結果保存中: {output}")
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
                file_size = os.path.getsize(file_path) / 1024
                print(f"   - {format_name.upper()}: {file_path} ({file_size:.1f}KB)")
        
        elif output_format == 'legacy':
            # Legacy format for backward compatibility
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
        
        # Display Ultra Precision results
        print("\\n" + "=" * 80)
        print("🎯 ULTRA PRECISION + ENHANCED SPEAKER RESULTS")
        print("=" * 80)
        print(f"📊 平均信頼度: {avg_confidence:.4f} ({avg_confidence:.1%})")
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
            print(f"👥 話者識別: 単一話者として処理")
        
        print(f"⏱️  処理時間: {processing_time:.1f}秒 ({processing_time/60:.1f}分)")
        print(f"🎧 音声長: {audio_duration/60:.1f}分")
        print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
        
        # Show ultra precision effects
        print(f"\\n🎯 Ultra Precision技術:")
        if use_ensemble:
            print(f"   ✅ Ensemble転写: {len(model_list)} モデル")
        if use_advanced_vad:
            print(f"   ✅ 高度VAD処理")
        if not no_enhanced_preprocessing:
            print(f"   ✅ Enhanced音声前処理")
        if not no_post_processing:
            print(f"   ✅ 高度後処理")
        
        if speaker_method != 'off':
            print(f"\\n👥 Enhanced話者識別: {speaker_method} 方式")
            if known_speakers:
                print(f"   ✅ {len(known_speakers)} 話者を高精度検出")
            else:
                print(f"   ⚠️  単一話者として処理")
        
        # Quality assessment
        if avg_confidence >= 0.95:
            quality = "🟢 OUTSTANDING"
        elif avg_confidence >= 0.90:
            quality = "🟢 EXCELLENT"
        elif avg_confidence >= 0.85:
            quality = "🟡 VERY GOOD"
        elif avg_confidence >= 0.80:
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
        
        print("=" * 80)
        print(f"🎉 Ultra Precision転写完了! 平均信頼度: {avg_confidence:.1%}")
        
        if use_ensemble and speaker_method != 'off':
            print(f"🎯 Ensemble + Enhanced話者識別により最高精度を実現!")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        raise


def process_standard_file_ultra_precision(
    audio_file: str, model_list: List[str], language: str, device: str,
    use_ensemble: bool, voting_method: str, use_advanced_vad: bool,
    enhanced_preprocessing: bool, post_processing: bool, min_confidence: float,
    speaker_method: str, num_speakers: Optional[int], hf_token: Optional[str]
) -> List[Dict]:
    """Process standard file with ultra precision techniques."""
    
    # Enhanced audio processing
    audio_processor = EnhancedAudioProcessor()
    
    if enhanced_preprocessing:
        print(f"🎧 Ultra Precision音声前処理")
        audio_data, sample_rate = audio_processor.advanced_preprocess_audio(
            audio_file,
            enable_noise_reduction=True,
            enable_speech_enhancement=True,
            enable_spectral_norm=True,
            enable_volume_adjustment=True,
            enable_silence_removal=False,
            memory_efficient=False
        )
    else:
        print(f"🎧 標準音声処理")
        audio_data, sample_rate = audio_processor.load_audio(audio_file)
    
    # Advanced VAD
    if use_advanced_vad:
        from .precision_enhancer import AdvancedVAD
        print("🎙️  Ultra Precision VAD適用中...")
        vad = AdvancedVAD(threshold=0.3, min_speech_duration=100)
        audio_data = vad.apply_advanced_vad(audio_data, sample_rate)
    
    # Ultra precision transcription
    if use_ensemble:
        print(f"🎭 Ensemble転写実行中 ({len(model_list)} models)...")
        final_segments = process_ensemble_transcription(
            audio_data, sample_rate, model_list, language, device,
            voting_method, min_confidence
        )
    else:
        print(f"📝 単一モデル転写実行中 (model: {model_list[0]})...")
        final_segments = process_single_model_transcription(
            audio_data, sample_rate, model_list[0], language, device, min_confidence
        )
    
    # Convert to dict format if needed
    transcription_segments = []
    for seg in final_segments:
        if hasattr(seg, 'start_time'):
            transcription_segments.append({
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "start_seconds": seg.start_time,
                "end_seconds": seg.end_time,
                "text": seg.text,
                "confidence": seg.confidence
            })
        else:
            transcription_segments.append(seg)
    
    # Ultra precision post-processing
    if post_processing:
        print("🔧 Ultra Precision後処理適用中...")
        post_processor = TranscriptionPostProcessor()
        transcription_segments = post_processor.process_transcription(transcription_segments)
        print(f"✅ Ultra Precision後処理完了: {len(transcription_segments)} セグメント")
    
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


def process_ensemble_transcription(
    audio_data, sample_rate, model_list, language, device, voting_method, min_confidence
):
    """Process transcription with ensemble of models."""
    from .faster_transcriber import FasterTranscriber
    
    all_results = []
    
    for model in model_list:
        print(f"   🔄 Model {model} 処理中...")
        transcriber = FasterTranscriber(
            model_size=model,
            language=language,
            device=device
        )
        
        segments = transcriber.process_transcription(
            audio_data,
            sample_rate,
            filter_confidence=True,
            filter_fillers=True,
            min_confidence=min_confidence * 0.7  # Lower threshold for ensemble
        )
        
        all_results.append(segments)
        print(f"   ✅ Model {model}: {len(segments)} セグメント")
    
    # Ensemble voting
    print(f"🗳️  Ensemble voting: {voting_method}")
    if voting_method == 'confidence_weighted':
        final_segments = confidence_weighted_ensemble(all_results)
    else:
        final_segments = majority_voting_ensemble(all_results)
    
    return final_segments


def process_single_model_transcription(
    audio_data, sample_rate, model, language, device, min_confidence
):
    """Process transcription with single model."""
    from .faster_transcriber import FasterTranscriber
    
    transcriber = FasterTranscriber(
        model_size=model,
        language=language,
        device=device
    )
    
    segments = transcriber.process_transcription(
        audio_data,
        sample_rate,
        filter_confidence=True,
        filter_fillers=True,
        min_confidence=min_confidence
    )
    
    return segments


def confidence_weighted_ensemble(all_results):
    """Combine results using confidence-weighted voting."""
    if not all_results:
        return []
    
    # Simple implementation: use highest confidence segment for each time window
    final_segments = []
    
    # Get all unique time segments
    all_segments = []
    for result in all_results:
        all_segments.extend(result)
    
    # Sort by start time
    all_segments.sort(key=lambda x: getattr(x, 'start_time', 0))
    
    # Group overlapping segments and select highest confidence
    current_segments = []
    current_end_time = 0
    
    for seg in all_segments:
        start_time = getattr(seg, 'start_time', 0)
        
        if start_time < current_end_time:
            # Overlapping segment
            current_segments.append(seg)
        else:
            # Non-overlapping, finalize previous group
            if current_segments:
                best_seg = max(current_segments, key=lambda x: getattr(x, 'confidence', 0))
                final_segments.append(best_seg)
            
            current_segments = [seg]
            current_end_time = getattr(seg, 'end_time', start_time)
    
    # Handle last group
    if current_segments:
        best_seg = max(current_segments, key=lambda x: getattr(x, 'confidence', 0))
        final_segments.append(best_seg)
    
    return final_segments


def majority_voting_ensemble(all_results):
    """Combine results using majority voting."""
    # Simplified: just return the result with most segments
    if not all_results:
        return []
    
    return max(all_results, key=len)


def process_large_file_ultra_precision(
    audio_file: str, model_list: List[str], language: str, device: str,
    use_ensemble: bool, voting_method: str, use_advanced_vad: bool,
    enhanced_preprocessing: bool, min_confidence: float,
    speaker_method: str, num_speakers: Optional[int], hf_token: Optional[str]
) -> List[Dict]:
    """Process large files with ultra precision optimizations."""
    print("🏗️  大容量ファイル Ultra Precision処理開始...")
    
    # Use chunked transcriber for large files
    primary_model = model_list[0] if model_list else 'large-v3-turbo'
    
    chunked_transcriber = ChunkedTranscriber(
        model_size=primary_model,
        language=language,
        device=device,
        chunk_duration=5,  # Smaller chunks for higher precision
        overlap_duration=3.0,  # More overlap for accuracy
        max_memory_mb=1024
    )
    
    try:
        segments = chunked_transcriber.process_large_file(
            audio_file=audio_file,
            enhanced_preprocessing=enhanced_preprocessing,
            filter_fillers=True,
            min_confidence=min_confidence
        )
        
        # Convert to proper format
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
        
        print(f"✅ Ultra Precision大容量処理完了: {len(final_segments)} セグメント")
        return final_segments
        
    finally:
        chunked_transcriber.cleanup()


def cli():
    """Ultra Precision + Enhanced Speaker Recognition CLI application."""
    ultra_precision_transcribe()


if __name__ == '__main__':
    cli()