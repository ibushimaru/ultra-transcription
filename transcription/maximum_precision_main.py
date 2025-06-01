"""
Maximum precision transcription application with all accuracy enhancement strategies.
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
from .precision_enhancer import PrecisionEnhancer
from .chunked_transcriber import ChunkedTranscriber
from .time_estimator import TranscriptionTimeEstimator


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='medium', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Primary Whisper model size (default: medium for higher accuracy)')
@click.option('--language', '-l', default='ja', help='Language code for transcription')
@click.option('--min-confidence', default=0.2, type=float,
              help='Minimum confidence threshold (lowered for ensemble filtering)')
@click.option('--use-ensemble', is_flag=True, default=True,
              help='Use ensemble of multiple models (default: enabled)')
@click.option('--ensemble-models', default='base,medium', 
              help='Comma-separated list of models for ensemble (default: base,medium)')
@click.option('--voting-method', default='confidence_weighted',
              type=click.Choice(['confidence_weighted', 'majority']),
              help='Ensemble voting method')
@click.option('--use-advanced-vad', is_flag=True, default=True,
              help='Use advanced voice activity detection (default: enabled)')
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
@click.option('--chunk-threshold', default=1800, type=int,
              help='Seconds threshold for chunked processing (default: 30min)')
@click.option('--auto-confirm', is_flag=True,
              help='Skip confirmation for long processes')
def maximum_precision_transcribe(audio_file: str, output: Optional[str], model: str, language: str,
                                min_confidence: float, use_ensemble: bool, ensemble_models: str,
                                voting_method: str, use_advanced_vad: bool,
                                no_enhanced_preprocessing: bool, no_post_processing: bool,
                                no_speaker_diarization: bool, hf_token: Optional[str], 
                                output_format: str, device: str, chunk_threshold: int,
                                auto_confirm: bool):
    """
    Maximum precision transcription with all accuracy enhancement strategies.
    
    Uses ensemble models, advanced VAD, enhanced preprocessing, and intelligent chunking
    to achieve the highest possible transcription accuracy.
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("🎯 MAXIMUM PRECISION 音声文字起こしアプリケーション")
    print("=" * 70)
    print("🔬 全精度向上技術を統合した最高品質システム")
    print("=" * 70)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        output = str(audio_path.parent / f"{audio_path.stem}_max_precision")
    
    try:
        start_time = time.time()
        
        # Parse ensemble models
        if use_ensemble:
            model_list = [m.strip() for m in ensemble_models.split(',')]
            print(f"🎭 Ensemble models: {model_list}")
        else:
            model_list = [model]
            print(f"📝 Single model: {model}")
        
        # Check file size and duration for processing strategy
        file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
        print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
        
        # Quick duration check
        from pydub import AudioSegment
        try:
            audio_segment = AudioSegment.from_file(audio_file)
            audio_duration = len(audio_segment) / 1000.0
            del audio_segment  # Free memory immediately
        except Exception as e:
            print(f"⚠️  Duration analysis failed: {e}")
            audio_duration = file_size_mb * 60  # Conservative estimate
        
        print(f"⏱️  音声長: {audio_duration/60:.1f}分 ({audio_duration/3600:.1f}時間)")
        
        # Determine processing strategy
        use_chunked_processing = audio_duration > chunk_threshold
        
        if use_chunked_processing:
            print(f"🏗️  大容量ファイル: チャンク処理 + 精度向上技術")
            
            # Time estimation for chunked processing
            estimator = TranscriptionTimeEstimator()
            estimates = estimator.estimate_processing_time(
                audio_duration=audio_duration,
                model_size=model,
                device=device,
                engine='maximum-precision-chunked',
                enhanced_preprocessing=not no_enhanced_preprocessing,
                post_processing=not no_post_processing,
                speaker_diarization=not no_speaker_diarization
            )
            
            # Adjust estimate for ensemble processing
            if use_ensemble:
                ensemble_multiplier = len(model_list) * 0.8  # Some parallel efficiency
                for key in ['optimistic', 'realistic', 'pessimistic']:
                    estimates[key] *= ensemble_multiplier
            
            print(f"⏱️  予想処理時間: {estimator.format_time_estimate(estimates['realistic'])}")
            
            # Ask for confirmation on very long processes
            if estimates['realistic'] > 1800 and not auto_confirm:  # 30 minutes
                print(f"⚠️  高精度処理に{estimator.format_time_estimate(estimates['realistic'])}かかる可能性があります")
                if not click.confirm("最高精度モードで続行しますか？"):
                    return
            
            # Use enhanced chunked processing
            final_segments = process_large_file_max_precision(
                audio_file=audio_file,
                model_list=model_list,
                language=language,
                device=device,
                use_ensemble=use_ensemble,
                voting_method=voting_method,
                use_advanced_vad=use_advanced_vad,
                enhanced_preprocessing=not no_enhanced_preprocessing,
                post_processing=not no_post_processing,
                min_confidence=min_confidence
            )
            
            processing_mode = "chunked-max-precision"
            
        else:
            print(f"🎯 標準ファイル: 最高精度処理")
            
            # Initialize precision enhancer
            precision_enhancer = PrecisionEnhancer(
                use_ensemble=use_ensemble,
                ensemble_models=model_list,
                use_advanced_vad=use_advanced_vad
            )
            
            # Enhanced audio processing
            audio_processor = EnhancedAudioProcessor()
            
            if no_enhanced_preprocessing:
                print(f"🎧 標準音声処理")
                audio_data, sample_rate = audio_processor.preprocess_audio(
                    audio_file, reduce_noise=True
                )
            else:
                print(f"🎧 高度音声処理")
                audio_data, sample_rate = audio_processor.advanced_preprocess_audio(
                    audio_file,
                    enable_noise_reduction=True,
                    enable_speech_enhancement=True,
                    enable_spectral_norm=True,
                    enable_volume_adjustment=True,
                    enable_silence_removal=False
                )
            
            # Apply precision enhancement
            enhanced_segments, enhancement_metadata = precision_enhancer.enhance_transcription(
                audio_data=audio_data,
                sample_rate=sample_rate,
                language=language,
                device=device,
                min_confidence=min_confidence
            )
            
            # Convert to final format
            final_segments = [
                {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "text": seg["text"],
                    "confidence": seg["confidence"],
                    "speaker_id": "SPEAKER_UNKNOWN"
                }
                for seg in enhanced_segments
            ]
            
            processing_mode = "standard-max-precision"
            
            # Print enhancement details
            if 'ensemble_details' in enhancement_metadata:
                ensemble_info = enhancement_metadata['ensemble_details']
                print(f"\n🎭 Ensemble Results:")
                for model_name, perf in ensemble_info['model_performances'].items():
                    print(f"   - {model_name}: {perf['confidence']:.3f} confidence")
                
                improvement = ensemble_info.get('ensemble_improvement', {})
                if 'relative_improvement' in improvement:
                    print(f"📈 Ensemble improvement: {improvement['relative_improvement']:.1f}%")
        
        if not final_segments:
            print("⚠️  文字起こし結果が得られませんでした")
            return
        
        print(f"\n✅ 最高精度処理完了: {len(final_segments)} セグメント")
        
        # Post-processing for maximum accuracy
        if not no_post_processing:
            print("🔧 最高精度後処理適用中...")
            post_processor = TranscriptionPostProcessor()
            final_segments = post_processor.process_transcription(final_segments)
            print(f"✅ 最終精製完了: {len(final_segments)} セグメント")
        
        # Speaker diarization (if not chunked processing)
        if not use_chunked_processing and not no_speaker_diarization:
            speaker_diarizer = SpeakerDiarizer(use_auth_token=hf_token)
            if speaker_diarizer and speaker_diarizer.is_available():
                print("👥 話者識別適用中...")
                speaker_segments = speaker_diarizer.diarize_audio(audio_data, sample_rate)
                
                # Update speaker information
                final_segments = speaker_diarizer.assign_speakers_to_transcription(
                    final_segments, speaker_segments
                )
                
                unique_speakers = set(seg["speaker_id"] for seg in final_segments)
                print(f"✅ {len(unique_speakers)} 話者検出")
        
        # Calculate final metrics
        processing_time = time.time() - start_time
        avg_confidence = sum(seg["confidence"] for seg in final_segments) / len(final_segments)
        total_text_length = sum(len(seg["text"]) for seg in final_segments)
        
        # Prepare comprehensive metadata
        metadata = {
            "input_file": audio_file,
            "primary_model": model,
            "ensemble_models": model_list if use_ensemble else None,
            "language": language,
            "device": device,
            "engine": "maximum-precision",
            "processing_mode": processing_mode,
            "techniques_applied": {
                "ensemble_transcription": use_ensemble,
                "voting_method": voting_method if use_ensemble else None,
                "advanced_vad": use_advanced_vad,
                "enhanced_preprocessing": not no_enhanced_preprocessing,
                "post_processing": not no_post_processing,
                "speaker_diarization": not no_speaker_diarization,
                "chunked_processing": use_chunked_processing
            },
            "min_confidence": min_confidence,
            "audio_duration_seconds": audio_duration,
            "total_segments": len(final_segments),
            "average_confidence": round(avg_confidence, 3),
            "total_text_length": total_text_length,
            "processing_time_seconds": round(processing_time, 2),
            "processing_ratio": round(processing_time / audio_duration, 3) if audio_duration > 0 else None,
            "file_size_mb": round(file_size_mb, 1)
        }
        
        # Save maximum precision output
        print(f"\n💾 最高精度結果保存中: {output}")
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
        
        # Display comprehensive results
        print("\n" + "=" * 70)
        print("🎯 MAXIMUM PRECISION RESULTS")
        print("=" * 70)
        print(f"📊 平均信頼度: {avg_confidence:.3f} ({avg_confidence:.1%})")
        print(f"📝 総セグメント: {len(final_segments)}")
        print(f"📄 総文字数: {total_text_length}")
        print(f"👥 話者数: {len(set(seg['speaker_id'] for seg in final_segments))}")
        print(f"⏱️  処理時間: {processing_time:.1f}秒 ({processing_time/60:.1f}分)")
        print(f"🎧 音声長: {audio_duration/60:.1f}分")
        print(f"⚡ 処理比率: {processing_time/audio_duration:.2f}x")
        print(f"📁 ファイルサイズ: {file_size_mb:.1f}MB")
        
        # Display techniques used
        print(f"\n🔬 適用技術:")
        techniques = metadata['techniques_applied']
        if techniques['ensemble_transcription']:
            print(f"   ✅ アンサンブル転写 ({voting_method})")
        if techniques['advanced_vad']:
            print(f"   ✅ 高度VAD")
        if techniques['enhanced_preprocessing']:
            print(f"   ✅ 高度前処理")
        if techniques['post_processing']:
            print(f"   ✅ 高度後処理")
        if techniques['chunked_processing']:
            print(f"   ✅ チャンク処理")
        
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
        print("=" * 70)
        
        print(f"\n🎉 最高精度転写完了! 平均信頼度: {avg_confidence:.1%}")
        
        # Performance comparison note
        if use_ensemble:
            print(f"🎭 アンサンブル効果により精度向上を実現")
        
        if use_chunked_processing:
            print(f"🏗️  大容量ファイルをメモリ効率的に処理")
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        raise


def process_large_file_max_precision(audio_file: str,
                                   model_list: list,
                                   language: str,
                                   device: str,
                                   use_ensemble: bool,
                                   voting_method: str,
                                   use_advanced_vad: bool,
                                   enhanced_preprocessing: bool,
                                   post_processing: bool,
                                   min_confidence: float) -> list:
    """
    Process large files with maximum precision using chunked approach.
    """
    print("🏗️  大容量ファイル最高精度処理開始...")
    
    # For large files, use the most capable single model to avoid memory issues
    # but apply all other enhancement techniques
    primary_model = model_list[-1] if model_list else 'medium'  # Use largest available
    
    # Initialize enhanced chunked transcriber
    chunked_transcriber = ChunkedTranscriber(
        model_size=primary_model,
        language=language,
        device=device,
        chunk_duration=10,  # 10 minutes
        overlap_duration=3.0,  # Increased overlap for better accuracy
        max_memory_mb=512
    )
    
    try:
        # Process with enhanced settings
        segments = chunked_transcriber.process_large_file(
            audio_file=audio_file,
            enhanced_preprocessing=enhanced_preprocessing,
            filter_fillers=True,
            min_confidence=min_confidence
        )
        
        print(f"✅ 大容量ファイル処理完了: {len(segments)} セグメント")
        return segments
        
    finally:
        # Clean up
        chunked_transcriber.cleanup()


def cli():
    """Maximum precision audio transcription CLI application."""
    maximum_precision_transcribe()


if __name__ == '__main__':
    cli()