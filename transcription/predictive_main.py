"""
Predictive CLI application with time estimation and progress tracking.
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
from .time_estimator import TranscriptionTimeEstimator
from .progress_tracker import ProgressTracker, EnhancedProgressCallback


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file base path (without extension)')
@click.option('--model', '-m', default='large-v3-turbo', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo']),
              help='Whisper model size (default: large-v3-turbo for maximum speed)')
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
@click.option('--show-progress', is_flag=True, default=True,
              help='Show real-time progress (default: enabled)')
@click.option('--confidence-level', default=0.8, type=float,
              help='Confidence level for time estimation (0.5-0.95)')
@click.option('--auto-confirm', is_flag=True,
              help='Skip confirmation for long processes')
def predictive_transcribe(audio_file: str, output: Optional[str], model: str, language: str,
                         min_confidence: float, no_enhanced_preprocessing: bool,
                         no_post_processing: bool, no_speaker_diarization: bool,
                         no_filter_fillers: bool, hf_token: Optional[str], 
                         output_format: str, device: str, show_progress: bool,
                         confidence_level: float, auto_confirm: bool):
    """
    Predictive transcription with time estimation and progress tracking.
    
    AUDIO_FILE: Path to the audio file (MP3, WAV, etc.)
    """
    
    print("🔮 PREDICTIVE 音声文字起こしアプリケーション")
    print("=" * 65)
    
    # Validate input file
    if not os.path.exists(audio_file):
        click.echo(f"Error: Audio file not found: {audio_file}", err=True)
        return
    
    # Set default output path if not provided
    if not output:
        audio_path = Path(audio_file)
        output = str(audio_path.parent / f"{audio_path.stem}_predictive")
    
    try:
        # Initialize time estimator
        estimator = TranscriptionTimeEstimator()
        
        # Get audio duration for estimation
        print("📏 音声ファイル分析中...")
        if no_enhanced_preprocessing:
            from .audio_processor import AudioProcessor
            temp_processor = AudioProcessor()
        else:
            temp_processor = EnhancedAudioProcessor()
        
        audio_duration = temp_processor.get_audio_duration(audio_file)
        print(f"🎵 検出された音声長: {estimator.format_time_estimate(audio_duration)}")
        
        # Generate time estimate
        estimates = estimator.estimate_processing_time(
            audio_duration=audio_duration,
            model_size=model,
            device=device,
            engine='ultra-enhanced' if not no_enhanced_preprocessing else 'faster-whisper',
            enhanced_preprocessing=not no_enhanced_preprocessing,
            post_processing=not no_post_processing,
            speaker_diarization=not no_speaker_diarization,
            confidence_level=confidence_level
        )
        
        # Display time estimation
        estimator.display_time_estimate(audio_duration, estimates)
        
        # Ask for confirmation on long processes
        if estimates['realistic'] > 300 and not auto_confirm:  # 5 minutes
            print(f"⚠️  処理に{estimator.format_time_estimate(estimates['realistic'])}かかる可能性があります")
            if not click.confirm("続行しますか？"):
                return
        
        # Initialize progress tracker
        tracker = None
        callback = None
        if show_progress:
            tracker = ProgressTracker(estimates['realistic'])
            callback = EnhancedProgressCallback(tracker)
            tracker.start()
        
        start_time = time.time()
        
        try:
            # Initialize components
            if callback:
                callback.audio_loading(0.1)
            
            print("\n🔧 コンポーネント初期化中...")
            
            if no_enhanced_preprocessing:
                from .audio_processor import AudioProcessor
                audio_processor = AudioProcessor()
            else:
                audio_processor = EnhancedAudioProcessor()
            
            transcriber = FasterTranscriber(model_size=model, language=language, device=device)
            
            if not no_post_processing:
                post_processor = TranscriptionPostProcessor()
            
            if not no_speaker_diarization:
                speaker_diarizer = SpeakerDiarizer(use_auth_token=hf_token)
            else:
                speaker_diarizer = None
            
            output_formatter = OutputFormatter()
            
            if callback:
                callback.audio_loading(1.0)
                tracker.complete_step("音声読み込み")
            
            # Enhanced audio processing
            if callback:
                callback.preprocessing("読み込み", 0.0)
            
            if no_enhanced_preprocessing:
                print("🎧 標準音声処理中...")
                audio_data, sample_rate = audio_processor.preprocess_audio(
                    audio_file, reduce_noise=True
                )
            else:
                print("🎧 高度音声処理中...")
                audio_data, sample_rate = audio_processor.advanced_preprocess_audio(
                    audio_file,
                    enable_noise_reduction=True,
                    enable_speech_enhancement=True,
                    enable_spectral_norm=False,
                    enable_volume_adjustment=True,
                    enable_silence_removal=False
                )
            
            if callback:
                callback.preprocessing("完了", 1.0)
                tracker.complete_step("前処理")
            
            # Transcription with progress updates
            if callback:
                callback.transcription(0.0)
            
            print(f"📝 文字起こし実行中 (model: {model}, device: {device})...")
            transcription_segments = transcriber.process_transcription(
                audio_data, 
                sample_rate,
                filter_confidence=True,
                filter_fillers=not no_filter_fillers,
                min_confidence=min_confidence
            )
            
            if callback:
                callback.transcription(1.0, len(transcription_segments), len(transcription_segments))
                tracker.complete_step("文字起こし")
            
            if not transcription_segments:
                print("⚠️  文字起こし結果が得られませんでした")
                return
            
            print(f"✅ {len(transcription_segments)} セグメント完了")
            
            # Convert to compatible format
            compatible_segments = []
            for seg in transcription_segments:
                compatible_segments.append({
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'text': seg.text,
                    'confidence': seg.confidence
                })
            
            # Post-processing
            if not no_post_processing:
                if callback:
                    callback.post_processing("テキスト修正", 0.5)
                
                print("🔧 後処理適用中...")
                compatible_segments = post_processor.process_transcription(compatible_segments)
                print(f"✅ {len(compatible_segments)} セグメントに精製")
                
                if callback:
                    callback.post_processing("完了", 1.0)
                    tracker.complete_step("後処理")
            
            # Speaker diarization
            if speaker_diarizer and speaker_diarizer.is_available():
                if callback:
                    callback.speaker_diarization(0.5)
                
                print("👥 スピーカー識別中...")
                speaker_segments = speaker_diarizer.diarize_audio(audio_data, sample_rate)
                
                final_segments = speaker_diarizer.assign_speakers_to_transcription(
                    compatible_segments, speaker_segments
                )
                
                unique_speakers = set(seg["speaker_id"] for seg in final_segments)
                print(f"✅ {len(unique_speakers)} 話者検出")
                
                if callback:
                    callback.speaker_diarization(1.0)
                    tracker.complete_step("スピーカー識別")
            else:
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
            
            # Save output
            if callback:
                callback.output_saving("", 0.5)
            
            print(f"💾 結果保存中: {output}")
            
            # Prepare metadata
            processing_time = time.time() - start_time
            avg_confidence = sum(seg["confidence"] for seg in final_segments) / len(final_segments)
            
            metadata = {
                "input_file": audio_file,
                "model_size": model,
                "language": language,
                "device": device,
                "engine": "predictive-enhanced",
                "processing_time_seconds": round(processing_time, 2),
                "audio_duration_seconds": audio_duration,
                "processing_ratio": round(processing_time / audio_duration, 3),
                "average_confidence": round(avg_confidence, 3),
                "estimated_time": round(estimates['realistic'], 2),
                "estimation_accuracy": round(abs(processing_time - estimates['realistic']) / estimates['realistic'], 3),
                "total_segments": len(final_segments)
            }
            
            if output_format == 'all':
                saved_files = output_formatter.save_all_formats(final_segments, output, metadata)
                if callback:
                    callback.output_saving("全形式", 1.0)
            else:
                output_data = output_formatter.prepare_output_data(final_segments)
                
                if output_format == 'json':
                    output_formatter.save_as_json(output_data, f"{output}.json", metadata)
                elif output_format == 'csv':
                    output_formatter.save_as_csv(output_data, f"{output}.csv")
                elif output_format == 'txt':
                    output_formatter.save_as_txt(output_data, f"{output}.txt")
                elif output_format == 'srt':
                    output_formatter.save_as_srt(output_data, f"{output}.srt")
                
                if callback:
                    callback.output_saving(output_format.upper(), 1.0)
            
            if callback:
                tracker.complete_step("出力保存")
            
        finally:
            if tracker:
                tracker.stop()
        
        # Final results
        processing_time = time.time() - start_time
        
        print("\n" + "=" * 65)
        print("🎯 PREDICTIVE RESULTS")
        print("=" * 65)
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
        print(f"📝 総セグメント: {len(final_segments)}")
        print(f"⚡ 処理速度: {processing_time/audio_duration:.2f}x")
        
        # Update estimator with actual performance
        from .time_estimator import ProcessingMetrics
        metrics = ProcessingMetrics(
            audio_duration=audio_duration,
            processing_time=processing_time,
            model_size=model,
            device=device,
            engine='predictive-enhanced',
            enhanced_preprocessing=not no_enhanced_preprocessing,
            post_processing=not no_post_processing,
            speaker_diarization=not no_speaker_diarization
        )
        estimator.track_actual_performance(metrics)
        
        print("=" * 65)
        print("🎉 予測的文字起こし完了!")
        
    except Exception as e:
        if 'tracker' in locals() and tracker:
            tracker.stop()
        print(f"❌ エラー発生: {e}")
        raise


def cli():
    """Predictive audio transcription CLI application."""
    predictive_transcribe()


if __name__ == '__main__':
    cli()