"""
Main CLI application for audio transcription.
"""

import click
import os
import time
from pathlib import Path
from typing import Optional

from .audio_processor import AudioProcessor
from .transcriber import Transcriber
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
def transcribe(audio_file: str, output: Optional[str], model: str, language: str,
               min_confidence: float, no_noise_reduction: bool, 
               no_speaker_diarization: bool, no_filter_fillers: bool,
               hf_token: Optional[str], output_format: str):
    """
    Transcribe audio file to text with speaker identification.
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("🎵 音声文字起こしアプリケーション")
    print("=" * 50)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        output = str(audio_path.parent / audio_path.stem)
    
    try:
        start_time = time.time()
        
        # Initialize components
        print("🔧 コンポーネントを初期化中...")
        audio_processor = AudioProcessor()
        transcriber = Transcriber(model_size=model, language=language)
        
        if not no_speaker_diarization:
            speaker_diarizer = SpeakerDiarizer(use_auth_token=hf_token)
        else:
            speaker_diarizer = None
        
        output_formatter = OutputFormatter()
        
        # Process audio
        print(f"🎧 音声ファイルを処理中: {audio_file}")
        audio_data, sample_rate = audio_processor.preprocess_audio(
            audio_file, 
            reduce_noise=not no_noise_reduction
        )
        
        # Get audio duration
        duration = audio_processor.get_audio_duration(audio_file)
        print(f"⏱️  音声長: {output_formatter.format_timestamp(duration)}")
        
        # Transcribe audio
        print(f"📝 文字起こし中 (モデル: {model})...")
        transcription_segments = transcriber.process_transcription(
            audio_data, 
            sample_rate,
            filter_confidence=True,
            filter_fillers=not no_filter_fillers,
            min_confidence=min_confidence
        )
        
        if not transcription_segments:
            print("⚠️  文字起こし結果が得られませんでした。")
            print("   - 音声ファイルの品質を確認してください")
            print("   - --min-confidence の値を下げてみてください")
            return
        
        print(f"✅ {len(transcription_segments)} セグメントの文字起こしが完了")
        
        # Speaker diarization
        if speaker_diarizer and speaker_diarizer.is_available():
            print("👥 スピーカー識別中...")
            speaker_segments = speaker_diarizer.diarize_audio(audio_data, sample_rate)
            
            # Combine transcription and speaker information
            final_segments = speaker_diarizer.assign_speakers_to_transcription(
                transcription_segments, speaker_segments
            )
            
            # Count unique speakers
            unique_speakers = set(seg["speaker_id"] for seg in final_segments)
            print(f"✅ {len(unique_speakers)} 人のスピーカーを検出")
        else:
            if speaker_diarizer is None:
                print("⏭️  スピーカー識別をスキップ")
            else:
                print("⚠️  スピーカー識別が利用できません")
            
            # Convert transcription segments to final format
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
            "min_confidence": min_confidence,
            "noise_reduction": not no_noise_reduction,
            "speaker_diarization": not no_speaker_diarization,
            "filler_filtering": not no_filter_fillers,
            "audio_duration_seconds": duration,
            "total_segments": len(final_segments),
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        # Save output
        print(f"💾 結果を保存中: {output}")
        
        if output_format == 'all':
            saved_files = output_formatter.save_all_formats(final_segments, output, metadata)
            print("📁 保存されたファイル:")
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
        print(f"\\n⏱️  総処理時間: {total_time:.1f}秒")
        print("🎉 文字起こしが完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        raise


def cli():
    """Audio transcription CLI application."""
    transcribe()


if __name__ == '__main__':
    cli()