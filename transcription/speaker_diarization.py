"""
Speaker diarization using pyannote.audio.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pyannote.audio import Pipeline
import tempfile
import soundfile as sf
import os


@dataclass
class SpeakerSegment:
    """Data class for speaker segment."""
    start_time: float
    end_time: float
    speaker_id: str


class SpeakerDiarizer:
    """Handle speaker diarization using pyannote.audio."""
    
    def __init__(self, use_auth_token: Optional[str] = None):
        """
        Initialize speaker diarizer.
        
        Args:
            use_auth_token: Hugging Face auth token for pyannote models
        """
        self.use_auth_token = use_auth_token
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load pyannote diarization pipeline."""
        try:
            print("Loading pyannote speaker diarization model...")
            
            # Try to load the pipeline
            if self.use_auth_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.use_auth_token
                )
            else:
                # Try without auth token first
                try:
                    self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                except Exception:
                    print("Warning: Could not load pyannote model without auth token.")
                    print("You may need to set up Hugging Face authentication.")
                    print("For now, speaker diarization will be disabled.")
                    self.pipeline = None
                    return
            
            # Set device
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                print("Using GPU for speaker diarization")
            else:
                print("Using CPU for speaker diarization")
            
            print("Speaker diarization model loaded successfully")
            
        except Exception as e:
            print(f"Error loading speaker diarization model: {e}")
            print("Speaker diarization will be disabled.")
            self.pipeline = None
    
    def diarize_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            List of speaker segments
        """
        if self.pipeline is None:
            print("Speaker diarization is not available.")
            return []
        
        # Create temporary audio file for pyannote
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)
        
        try:
            # Perform diarization
            diarization = self.pipeline(temp_path)
            
            # Convert to speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(SpeakerSegment(
                    start_time=turn.start,
                    end_time=turn.end,
                    speaker_id=speaker
                ))
            
            return segments
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def assign_speakers_to_transcription(self, 
                                       transcription_segments: List,
                                       speaker_segments: List[SpeakerSegment]) -> List[Dict]:
        """
        Assign speaker IDs to transcription segments.
        
        Args:
            transcription_segments: List of transcription segments
            speaker_segments: List of speaker segments
            
        Returns:
            List of transcription segments with speaker IDs
        """
        if not speaker_segments:
            # No speaker information available
            return [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "text": seg.text,
                    "confidence": seg.confidence,
                    "speaker_id": "SPEAKER_UNKNOWN"
                }
                for seg in transcription_segments
            ]
        
        # Create speaker timeline for fast lookup
        speaker_timeline = []
        for speaker_seg in speaker_segments:
            speaker_timeline.append((speaker_seg.start_time, speaker_seg.end_time, speaker_seg.speaker_id))
        
        # Sort by start time
        speaker_timeline.sort(key=lambda x: x[0])
        
        # Assign speakers to transcription segments
        result = []
        for trans_seg in transcription_segments:
            # Find overlapping speaker segment
            assigned_speaker = "SPEAKER_UNKNOWN"
            max_overlap = 0.0
            
            trans_start = trans_seg.start_time
            trans_end = trans_seg.end_time
            trans_duration = trans_end - trans_start
            
            for speaker_start, speaker_end, speaker_id in speaker_timeline:
                # Calculate overlap
                overlap_start = max(trans_start, speaker_start)
                overlap_end = min(trans_end, speaker_end)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlap_ratio = overlap_duration / trans_duration
                    
                    if overlap_ratio > max_overlap:
                        max_overlap = overlap_ratio
                        assigned_speaker = speaker_id
            
            result.append({
                "start_time": trans_seg.start_time,
                "end_time": trans_seg.end_time,
                "text": trans_seg.text,
                "confidence": trans_seg.confidence,
                "speaker_id": assigned_speaker
            })
        
        return result
    
    def is_available(self) -> bool:
        """Check if speaker diarization is available."""
        return self.pipeline is not None