"""
Complete transcription pipeline orchestrator.
"""

from typing import List, Dict, Optional
import time

from .audio_processor import AudioProcessor
from .transcriber import Transcriber, TranscriptionSegment
from .speaker_diarization import SpeakerDiarizer
from .output_formatter import OutputFormatter


class TranscriptionPipeline:
    """
    Complete pipeline for audio transcription with speaker diarization.
    """
    
    def __init__(self, 
                 whisper_model: str = "base",
                 language: str = "ja",
                 hf_token: Optional[str] = None):
        """
        Initialize transcription pipeline.
        
        Args:
            whisper_model: Whisper model size
            language: Language code for transcription
            hf_token: Hugging Face token for pyannote models
        """
        self.audio_processor = AudioProcessor()
        self.transcriber = Transcriber(model_size=whisper_model, language=language)
        self.speaker_diarizer = SpeakerDiarizer(use_auth_token=hf_token)
        self.output_formatter = OutputFormatter()
        
        self.processing_stats = {}
    
    def process_file(self, 
                    input_path: str,
                    output_path: str,
                    enable_noise_reduction: bool = True,
                    enable_speaker_diarization: bool = True,
                    enable_filler_filtering: bool = True,
                    min_confidence: float = 0.3,
                    output_formats: List[str] = None) -> Dict:
        """
        Process audio file through complete pipeline.
        
        Args:
            input_path: Path to input audio file
            output_path: Base path for output files
            enable_noise_reduction: Whether to apply noise reduction
            enable_speaker_diarization: Whether to perform speaker diarization
            enable_filler_filtering: Whether to filter filler words
            min_confidence: Minimum confidence threshold
            output_formats: List of output formats to generate
            
        Returns:
            Dictionary with processing results and statistics
        """
        if output_formats is None:
            output_formats = ['json', 'csv', 'txt', 'srt']
        
        start_time = time.time()
        
        # Step 1: Load and preprocess audio
        print("Loading and preprocessing audio...")
        step_start = time.time()
        
        audio_data, sample_rate = self.audio_processor.preprocess_audio(
            input_path, 
            reduce_noise=enable_noise_reduction
        )
        
        duration = self.audio_processor.get_audio_duration(input_path)
        
        self.processing_stats['audio_loading_time'] = time.time() - step_start
        
        # Step 2: Transcription
        print("Performing transcription...")
        step_start = time.time()
        
        transcription_segments = self.transcriber.process_transcription(
            audio_data,
            sample_rate,
            filter_confidence=True,
            filter_fillers=enable_filler_filtering,
            min_confidence=min_confidence
        )
        
        self.processing_stats['transcription_time'] = time.time() - step_start
        
        if not transcription_segments:
            return {
                'success': False,
                'error': 'No transcription segments generated',
                'stats': self.processing_stats
            }
        
        # Step 3: Speaker diarization (if enabled)
        speaker_segments = []
        if enable_speaker_diarization and self.speaker_diarizer.is_available():
            print("Performing speaker diarization...")
            step_start = time.time()
            
            speaker_segments = self.speaker_diarizer.diarize_audio(audio_data, sample_rate)
            
            self.processing_stats['diarization_time'] = time.time() - step_start
        
        # Step 4: Combine transcription and speaker information
        print("Combining transcription and speaker information...")
        
        if speaker_segments:
            final_segments = self.speaker_diarizer.assign_speakers_to_transcription(
                transcription_segments, speaker_segments
            )
        else:
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
        
        # Step 5: Generate outputs
        print("Generating output files...")
        step_start = time.time()
        
        metadata = {
            "input_file": input_path,
            "audio_duration_seconds": duration,
            "total_segments": len(final_segments),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "settings": {
                "noise_reduction": enable_noise_reduction,
                "speaker_diarization": enable_speaker_diarization,
                "filler_filtering": enable_filler_filtering,
                "min_confidence": min_confidence
            }
        }
        
        # Save in requested formats
        saved_files = {}
        output_data = self.output_formatter.prepare_output_data(final_segments)
        
        for format_type in output_formats:
            if format_type == 'json':
                path = f"{output_path}.json"
                self.output_formatter.save_as_json(output_data, path, metadata)
                saved_files['json'] = path
            elif format_type == 'csv':
                path = f"{output_path}.csv"
                self.output_formatter.save_as_csv(output_data, path)
                saved_files['csv'] = path
            elif format_type == 'txt':
                path = f"{output_path}.txt"
                self.output_formatter.save_as_txt(output_data, path)
                saved_files['txt'] = path
            elif format_type == 'srt':
                path = f"{output_path}.srt"
                self.output_formatter.save_as_srt(output_data, path)
                saved_files['srt'] = path
        
        self.processing_stats['output_generation_time'] = time.time() - step_start
        self.processing_stats['total_time'] = time.time() - start_time
        
        # Calculate statistics
        unique_speakers = set(seg["speaker_id"] for seg in final_segments)
        avg_confidence = sum(seg["confidence"] for seg in final_segments) / len(final_segments)
        
        return {
            'success': True,
            'segments': final_segments,
            'saved_files': saved_files,
            'stats': {
                **self.processing_stats,
                'total_segments': len(final_segments),
                'unique_speakers': len(unique_speakers),
                'average_confidence': round(avg_confidence, 3),
                'audio_duration': duration,
                'speaker_ids': list(unique_speakers)
            },
            'metadata': metadata
        }