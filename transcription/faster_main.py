"""
Enhanced CLI application using faster-whisper for audio transcription.
"""

import click
import os
import time
from pathlib import Path
from typing import Optional

from .audio_processor import AudioProcessor
from .faster_transcriber import FasterTranscriber
from .speaker_diarization import SpeakerDiarizer
from .output_formatter import OutputFormatter


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='base', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
              help='Whisper model size')
@click.option('--language', '-l', default='ja', help='Language code for transcription')
@click.option('--min-confidence', default=0.3, type=float,
              help='Minimum confidence threshold for segments')
@click.option('--no-noise-reduction', is_flag=True, 
              help='Skip noise reduction preprocessing')
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
def transcribe_faster(audio_file: str, output: Optional[str], model: str, language: str,
                     min_confidence: float, no_noise_reduction: bool, 
                     no_speaker_diarization: bool, no_filter_fillers: bool,
                     hf_token: Optional[str], output_format: str, device: str):
    """
    Enhanced transcribe audio file using faster-whisper.
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("🚀 Enhanced 音声文字起こしアプリケーション (faster-whisper)")
    print("=" * 60)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        output = str(audio_path.parent / f"{audio_path.stem}_faster")
    
    try:
        start_time = time.time()
        
        # Initialize components
        print("🔧 Enhanced components initializing...")
        audio_processor = AudioProcessor()
        transcriber = FasterTranscriber(model_size=model, language=language, device=device)
        
        if not no_speaker_diarization:
            speaker_diarizer = SpeakerDiarizer(use_auth_token=hf_token)
        else:
            speaker_diarizer = None
        
        output_formatter = OutputFormatter()
        
        # Process audio
        print(f"🎧 Processing audio file: {audio_file}")
        audio_data, sample_rate = audio_processor.preprocess_audio(
            audio_file, 
            reduce_noise=not no_noise_reduction
        )
        
        # Get audio duration
        duration = audio_processor.get_audio_duration(audio_file)
        print(f"⏱️  Audio duration: {output_formatter.format_timestamp(duration)}")
        
        # Transcribe audio
        print(f"📝 Enhanced transcription (model: {model}, device: {device})...")
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
        
        print(f"✅ {len(transcription_segments)} segments transcribed successfully")
        
        # Convert to compatible format for speaker diarization
        compatible_segments = []
        for seg in transcription_segments:
            compatible_segments.append(type('obj', (object,), {
                'start_time': seg.start_time,
                'end_time': seg.end_time,
                'text': seg.text,
                'confidence': seg.confidence
            }))
        
        # Speaker diarization
        if speaker_diarizer and speaker_diarizer.is_available():
            print("👥 Speaker identification...")
            speaker_segments = speaker_diarizer.diarize_audio(audio_data, sample_rate)
            
            # Combine transcription and speaker information
            final_segments = speaker_diarizer.assign_speakers_to_transcription(
                compatible_segments, speaker_segments
            )
            
            # Count unique speakers
            unique_speakers = set(seg["speaker_id"] for seg in final_segments)
            print(f"✅ {len(unique_speakers)} speakers detected")
        else:
            if speaker_diarizer is None:
                print("⏭️  Speaker identification skipped")
            else:
                print("⚠️  Speaker identification not available")
            
            # Convert faster transcription segments to final format
            final_segments = [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "confidence": seg.confidence,
                    "speaker_id": "SPEAKER_UNKNOWN"
                }
                for seg in transcription_segments
            ]
        
        # Prepare metadata
        metadata = {
            "input_file": audio_file,
            "model_size": model,
            "language": language,
            "device": device,
            "engine": "faster-whisper",
            "min_confidence": min_confidence,
            "noise_reduction": not no_noise_reduction,
            "speaker_diarization": not no_speaker_diarization,
            "filler_filtering": not no_filter_fillers,
            "audio_duration_seconds": duration,
            "total_segments": len(final_segments),
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        # Save output
        print(f"💾 Saving results: {output}")
        
        if output_format == 'all':
            saved_files = output_formatter.save_all_formats(final_segments, output, metadata)
            print("📁 Saved files:")
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
        
        # Display summary
        output_formatter.print_summary(final_segments, duration)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total processing time: {total_time:.1f} seconds")
        print("🎉 Enhanced transcription completed!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        raise


def cli():
    """Enhanced audio transcription CLI application."""
    transcribe_faster()


if __name__ == '__main__':
    cli()