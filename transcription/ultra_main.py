"""
Ultra-enhanced CLI application with maximum accuracy features.
"""

import click
import os
import time
from pathlib import Path
from typing import Optional

from .enhanced_audio_processor import EnhancedAudioProcessor
from .faster_transcriber import FasterTranscriber
from .post_processor import TranscriptionPostProcessor
from .speaker_diarization import SpeakerDiarizer
from .output_formatter import OutputFormatter
from .chunked_transcriber import ChunkedTranscriber
from .time_estimator import TranscriptionTimeEstimator


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='base', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--language', '-l', default='ja', help='Language code for transcription')
@click.option('--min-confidence', default=0.3, type=float,
              help='Minimum confidence threshold for segments')
@click.option('--no-enhanced-preprocessing', is_flag=True,
              help='Skip enhanced audio preprocessing')
@click.option('--no-post-processing', is_flag=True,
              help='Skip post-processing corrections')
@click.option('--no-speaker-diarization', is_flag=True,
              help='Skip speaker diarization')
@click.option('--no-filter-fillers', is_flag=True,
              help='Skip filler word filtering')
@click.option('--hf-token', help='Hugging Face token for pyannote models')
@click.option('--format', 'output_format', 
              type=click.Choice(['all', 'json', 'csv', 'txt', 'srt']),
              default='all', help='Output format')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']),
              help='Device to run on (cpu, cuda)')
@click.option('--enable-spectral-norm', is_flag=True,
              help='Enable spectral normalization')
@click.option('--enable-silence-removal', is_flag=True,
              help='Enable intelligent silence removal')
@click.option('--force-chunked', is_flag=True,
              help='Force chunked processing for large files')
@click.option('--chunk-duration', default=10, type=int,
              help='Chunk duration in minutes for large files (default: 10)')
@click.option('--max-memory', default=512, type=int,
              help='Maximum memory per chunk in MB (default: 512)')
@click.option('--auto-confirm', is_flag=True,
              help='Skip confirmation for long processes')
def ultra_transcribe(audio_file: str, output: Optional[str], model: str, language: str,
                    min_confidence: float, no_enhanced_preprocessing: bool,
                    no_post_processing: bool, no_speaker_diarization: bool,
                    no_filter_fillers: bool, hf_token: Optional[str], 
                    output_format: str, device: str, enable_spectral_norm: bool,
                    enable_silence_removal: bool, force_chunked: bool,
                    chunk_duration: int, max_memory: int, auto_confirm: bool):
    """
    Ultra-enhanced transcription with maximum accuracy features.
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("🚀 ULTRA Enhanced 音声文字起こしアプリケーション")
    print("=" * 65)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        output = str(audio_path.parent / f"{audio_path.stem}_ultra")
    
    try:
        start_time = time.time()
        
        # Check file size and duration for large file detection
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
        
        print(f"⏱️  推定音声長: {audio_duration/60:.1f}分 ({audio_duration/3600:.1f}時間)")
        
        # Automatic large file detection (30+ minutes or force_chunked)
        use_chunked_processing = force_chunked or audio_duration > 1800  # 30 minutes
        
        if use_chunked_processing:
            print(f"🏗️  大容量ファイル検出: チャンク処理モードに切り替え")
            print(f"📦 チャンク設定: {chunk_duration}分チャンク, {max_memory}MB制限")
            
            # Time estimation for chunked processing
            estimator = TranscriptionTimeEstimator()
            estimates = estimator.estimate_processing_time(
                audio_duration=audio_duration,
                model_size=model,
                device=device,
                engine='chunked-enhanced',
                enhanced_preprocessing=not no_enhanced_preprocessing,
                post_processing=not no_post_processing,
                speaker_diarization=not no_speaker_diarization
            )
            
            print(f"⏱️  予想処理時間: {estimator.format_time_estimate(estimates['realistic'])}")
            
            # Ask for confirmation on long processes
            if estimates['realistic'] > 600 and not auto_confirm:  # 10 minutes
                print(f"⚠️  処理に{estimator.format_time_estimate(estimates['realistic'])}かかる可能性があります")
                if not click.confirm("チャンク処理で続行しますか？"):
                    return
            
            # Initialize chunked transcriber
            chunked_transcriber = ChunkedTranscriber(
                model_size=model,
                language=language,
                device=device,
                chunk_duration=chunk_duration,
                overlap_duration=2.0,
                max_memory_mb=max_memory
            )
            
            try:
                # Process with chunked approach
                print("🚀 Ultra+チャンク処理開始...")
                segments = chunked_transcriber.process_large_file(
                    audio_file=audio_file,
                    enhanced_preprocessing=not no_enhanced_preprocessing,
                    filter_fillers=not no_filter_fillers,
                    min_confidence=min_confidence
                )
                
                if not segments:
                    print("⚠️  文字起こし結果が得られませんでした")
                    return
                
                print(f"✅ チャンク処理完了: {len(segments)} セグメント")
                
                # Convert to compatible format (already in correct format)
                final_segments = segments
                
            finally:
                # Clean up chunked transcriber resources
                chunked_transcriber.cleanup()
                
        else:
            print("🔧 Ultra components initializing...")
            
            if no_enhanced_preprocessing:
                from .audio_processor import AudioProcessor
                audio_processor = AudioProcessor()
            else:
                audio_processor = EnhancedAudioProcessor()
            
            transcriber = FasterTranscriber(model_size=model, language=language, device=device)
            output_formatter = OutputFormatter()
            
            # Enhanced audio processing
            if no_enhanced_preprocessing:
                print(f"🎧 Standard audio processing: {audio_file}")
                audio_data, sample_rate = audio_processor.preprocess_audio(
                    audio_file, reduce_noise=True
                )
            else:
                print(f"🎧 Ultra audio processing: {audio_file}")
                audio_data, sample_rate = audio_processor.advanced_preprocess_audio(
                    audio_file,
                    enable_noise_reduction=True,
                    enable_speech_enhancement=True,
                    enable_spectral_norm=enable_spectral_norm,
                    enable_volume_adjustment=True,
                    enable_silence_removal=enable_silence_removal
                )
            
            # Standard ultra transcription
            print(f"📝 Ultra transcription (model: {model}, device: {device})...")
            transcription_segments = transcriber.process_transcription(
                audio_data, 
                sample_rate,
                filter_confidence=True,
                filter_fillers=not no_filter_fillers,
                min_confidence=min_confidence
            )
            
            if not transcription_segments:
                print("⚠️  No transcription results obtained.")
                print("   - Check audio file quality")
                print("   - Try lowering --min-confidence value")
                return
            
            print(f"✅ {len(transcription_segments)} segments transcribed")
            
            # Convert to compatible format
            compatible_segments = []
            for seg in transcription_segments:
                compatible_segments.append({
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'text': seg.text,
                    'confidence': seg.confidence
                })
            
            # Speaker diarization (for standard processing only)
            if not no_speaker_diarization:
                speaker_diarizer = SpeakerDiarizer(use_auth_token=hf_token)
                if speaker_diarizer and speaker_diarizer.is_available():
                    print("👥 Ultra speaker identification...")
                    speaker_segments = speaker_diarizer.diarize_audio(audio_data, sample_rate)
                    
                    # Combine transcription and speaker information
                    final_segments = speaker_diarizer.assign_speakers_to_transcription(
                        compatible_segments, speaker_segments
                    )
                    
                    # Count unique speakers
                    unique_speakers = set(seg["speaker_id"] for seg in final_segments)
                    print(f"✅ {len(unique_speakers)} speakers detected")
                else:
                    print("⚠️  Speaker identification not available")
                    # Convert to final format
                    final_segments = [
                        {
                            "start_time": seg["start_time"],
                            "end_time": seg["end_time"],
                            "text": seg["text"],
                            "confidence": seg["confidence"],
                            "speaker_id": "SPEAKER_UNKNOWN"
                        }
                        for seg in compatible_segments
                    ]
            else:
                print("⏭️  Speaker identification skipped")
                # Convert to final format
                final_segments = [
                    {
                        "start_time": seg["start_time"],
                        "end_time": seg["end_time"],
                        "text": seg["text"],
                        "confidence": seg["confidence"],
                        "speaker_id": "SPEAKER_UNKNOWN"
                    }
                    for seg in compatible_segments
                ]
        
        # Post-processing for accuracy improvement (both modes)
        if not no_post_processing:
            print("🔧 Applying ultra post-processing...")
            post_processor = TranscriptionPostProcessor()
            final_segments = post_processor.process_transcription(final_segments)
            print(f"✅ Post-processed to {len(final_segments)} refined segments")
        
        # Calculate enhanced metrics
        processing_time = time.time() - start_time
        avg_confidence = sum(seg["confidence"] for seg in final_segments) / len(final_segments)
        total_text_length = sum(len(seg["text"]) for seg in final_segments)
        
        # Prepare enhanced metadata
        metadata = {
            "input_file": audio_file,
            "model_size": model,
            "language": language,
            "device": device,
            "engine": "ultra-enhanced" + ("-chunked" if use_chunked_processing else ""),
            "min_confidence": min_confidence,
            "enhanced_preprocessing": not no_enhanced_preprocessing,
            "post_processing": not no_post_processing,
            "speaker_diarization": not no_speaker_diarization,
            "filler_filtering": not no_filter_fillers,
            "spectral_normalization": enable_spectral_norm if not use_chunked_processing else False,
            "silence_removal": enable_silence_removal if not use_chunked_processing else False,
            "chunked_processing": use_chunked_processing,
            "chunk_duration_minutes": chunk_duration if use_chunked_processing else None,
            "max_memory_mb": max_memory if use_chunked_processing else None,
            "audio_duration_seconds": audio_duration,
            "total_segments": len(final_segments),
            "average_confidence": round(avg_confidence, 3),
            "total_text_length": total_text_length,
            "processing_time_seconds": round(processing_time, 2),
            "processing_ratio": round(processing_time / audio_duration, 3) if audio_duration > 0 else None,
            "file_size_mb": round(file_size_mb, 1)
        }
        
        # Save ultra output
        print(f"💾 Saving ultra results: {output}")
        output_formatter = OutputFormatter()
        
        if output_format == 'all':
            saved_files = output_formatter.save_all_formats(final_segments, output, metadata)
            print("📁 Ultra saved files:")
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
        
        # Display ultra summary
        print("\n" + "=" * 65)
        print("🎯 ULTRA TRANSCRIPTION RESULTS")
        if use_chunked_processing:
            print("🏗️  (チャンク処理モード)")
        print("=" * 65)
        print(f"📊 Average Confidence: {avg_confidence:.3f}")
        print(f"📝 Total Segments: {len(final_segments)}")
        print(f"📄 Total Characters: {total_text_length}")
        print(f"👥 Speakers Detected: {len(set(seg['speaker_id'] for seg in final_segments))}")
        print(f"⏱️  Processing Time: {processing_time:.1f}s")
        print(f"🎧 Audio Duration: {audio_duration/60:.1f}min")
        print(f"⚡ Processing Ratio: {processing_time/audio_duration:.2f}x")
        print(f"📁 File Size: {file_size_mb:.1f}MB")
        
        if use_chunked_processing:
            print(f"📦 Chunks Used: {chunk_duration}min chunks")
            print(f"🧠 Memory Limit: {max_memory}MB/chunk")
        
        # Quality assessment
        if avg_confidence >= 0.85:
            quality = "🟢 EXCELLENT"
        elif avg_confidence >= 0.75:
            quality = "🟡 VERY GOOD"
        elif avg_confidence >= 0.65:
            quality = "🟠 GOOD"
        else:
            quality = "🔴 FAIR"
        
        print(f"🎯 Quality Assessment: {quality}")
        print("=" * 65)
        
        print(f"\n🎉 ULTRA transcription completed with {avg_confidence:.1%} average confidence!")
        if use_chunked_processing:
            print(f"🏗️  Large file processing successful with memory-efficient chunking!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise


def cli():
    """Ultra-enhanced audio transcription CLI application."""
    ultra_transcribe()


if __name__ == '__main__':
    cli()