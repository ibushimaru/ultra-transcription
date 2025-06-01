"""
Speech-to-text transcription using OpenAI Whisper.
"""

import whisper
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class TranscriptionSegment:
    """Data class for transcription segment."""
    start_time: float
    end_time: float
    text: str
    confidence: float
    no_speech_prob: float


class Transcriber:
    """Handle speech-to-text transcription using Whisper."""
    
    def __init__(self, model_size: str = "base", language: str = "ja"):
        """
        Initialize transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code for transcription
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        print(f"Loading Whisper model: {self.model_size}")
        self.model = whisper.load_model(self.model_size)
        print("Model loaded successfully")
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[TranscriptionSegment]:
        """
        Transcribe audio data to text with timestamps.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            
        Returns:
            List of transcription segments
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure audio is in the right format for Whisper
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Transcribe with segment-level timestamps
        result = self.model.transcribe(
            audio_data,
            language=self.language,
            word_timestamps=False,
            verbose=True
        )
        
        print(f"Transcription result: {len(result.get('segments', []))} segments found")
        print(f"Full result keys: {result.keys()}")
        if result.get('segments'):
            print(f"First segment: {result['segments'][0]}")
        else:
            print(f"No segments found. Text: '{result.get('text', '')}'")
            print(f"Language: {result.get('language', 'unknown')}")
        
        segments = []
        for segment in result["segments"]:
            # Calculate confidence from avg_logprob (more reliable for segment-level)
            confidence = 0.0
            if "avg_logprob" in segment:
                # Convert logprob to confidence score (0-1 range)
                # avg_logprob typically ranges from -1 to 0
                confidence = max(0.0, min(1.0, segment["avg_logprob"] + 1.0))
            elif "words" in segment and segment["words"]:
                confidences = [word.get("probability", 0.0) for word in segment["words"] if "probability" in word]
                confidence = np.mean(confidences) if confidences else 0.5
            else:
                # Default moderate confidence if no data available
                confidence = 0.5
            
            # Get no_speech_probability
            no_speech_prob = segment.get("no_speech_prob", 0.0)
            
            segments.append(TranscriptionSegment(
                start_time=segment["start"],
                end_time=segment["end"],
                text=segment["text"].strip(),
                confidence=confidence,
                no_speech_prob=no_speech_prob
            ))
        
        return segments
    
    def filter_low_confidence_segments(self, segments: List[TranscriptionSegment], 
                                     min_confidence: float = 0.3) -> List[TranscriptionSegment]:
        """
        Filter out segments with low confidence.
        
        Args:
            segments: List of transcription segments
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered segments
        """
        return [seg for seg in segments if seg.confidence >= min_confidence]
    
    def filter_filler_words(self, segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
        """
        Remove filler words and meaningless sounds.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Filtered segments
        """
        # Japanese filler words and sounds
        filler_patterns = [
            r'^\s*[えーあーうーおー]+\s*$',  # えー、あー、うー、おー
            r'^\s*[エーアーウーオー]+\s*$',  # エー、アー、ウー、オー
            r'^\s*ん+\s*$',                  # ん、んん、んんん
            r'^\s*はい\s*$',                 # はい (単独)
            r'^\s*そう\s*$',                 # そう (単独)
            r'^\s*ま[あー]*\s*$',           # まあ、まー
            r'^\s*え[えー]*と?\s*$',         # ええと、えーと
            r'^\s*あの[ー～]*\s*$',          # あの、あのー
            r'^\s*その[ー～]*\s*$',          # その、そのー
            r'^\s*[。、！？!?]+\s*$',        # 記号のみ
        ]
        
        filtered_segments = []
        for segment in segments:
            text = segment.text.strip()
            
            # Skip empty text
            if not text:
                continue
            
            # Check against filler patterns
            is_filler = any(re.match(pattern, text, re.IGNORECASE) for pattern in filler_patterns)
            
            # Skip segments with high no_speech_prob
            if segment.no_speech_prob > 0.8:
                continue
            
            if not is_filler:
                filtered_segments.append(segment)
        
        return filtered_segments
    
    def process_transcription(self, audio_data: np.ndarray, sample_rate: int = 16000,
                            filter_confidence: bool = True, filter_fillers: bool = True,
                            min_confidence: float = 0.3) -> List[TranscriptionSegment]:
        """
        Complete transcription processing pipeline.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            filter_confidence: Whether to filter low confidence segments
            filter_fillers: Whether to filter filler words
            min_confidence: Minimum confidence threshold
            
        Returns:
            Processed transcription segments
        """
        # Transcribe audio
        segments = self.transcribe_audio(audio_data, sample_rate)
        
        # Apply filters
        if filter_confidence:
            segments = self.filter_low_confidence_segments(segments, min_confidence)
        
        if filter_fillers:
            segments = self.filter_filler_words(segments)
        
        return segments