"""
Enhanced speech-to-text transcription using faster-whisper.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from faster_whisper import WhisperModel


@dataclass
class FasterTranscriptionSegment:
    """Data class for faster transcription segment."""
    start_time: float
    end_time: float
    text: str
    confidence: float
    no_speech_prob: float


class FasterTranscriber:
    """Handle speech-to-text transcription using faster-whisper."""
    
    def __init__(self, model_size: str = "base", language: str = "ja", device: str = "cpu"):
        """
        Initialize faster transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Language code for transcription
            device: Device to run on (cpu, cuda)
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load faster-whisper model."""
        print(f"Loading faster-whisper model: {self.model_size}")
        # Use compute_type="int8" for better performance on CPU
        compute_type = "int8" if self.device == "cpu" else "float16"
        self.model = WhisperModel(
            self.model_size, 
            device=self.device,
            compute_type=compute_type
        )
        print("Faster-whisper model loaded successfully")
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[FasterTranscriptionSegment]:
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
        
        # Ensure audio is in the right format for faster-whisper
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(
            audio_data,
            language=self.language,
            word_timestamps=True,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"Faster-whisper transcription info: {info}")
        
        result_segments = []
        for segment in segments:
            # Calculate confidence from word-level probabilities
            confidence = 0.0
            if hasattr(segment, 'words') and segment.words:
                word_probs = [word.probability for word in segment.words if hasattr(word, 'probability')]
                confidence = np.mean(word_probs) if word_probs else 0.5
            else:
                confidence = 0.5  # Default confidence
            
            # Get no_speech_probability (if available)
            no_speech_prob = getattr(segment, 'no_speech_prob', 0.0)
            
            result_segments.append(FasterTranscriptionSegment(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text.strip(),
                confidence=confidence,
                no_speech_prob=no_speech_prob
            ))
        
        return result_segments
    
    def filter_low_confidence_segments(self, segments: List[FasterTranscriptionSegment], 
                                     min_confidence: float = 0.3) -> List[FasterTranscriptionSegment]:
        """
        Filter out segments with low confidence.
        
        Args:
            segments: List of transcription segments
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered segments
        """
        return [seg for seg in segments if seg.confidence >= min_confidence]
    
    def filter_filler_words(self, segments: List[FasterTranscriptionSegment]) -> List[FasterTranscriptionSegment]:
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
            r'^\s*よいしょ\s*$',             # よいしょ
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
            
            # Skip very short segments that are likely noise
            if len(text) <= 2 and segment.end_time - segment.start_time < 1.0:
                continue
            
            if not is_filler:
                filtered_segments.append(segment)
        
        return filtered_segments
    
    def process_transcription(self, audio_data: np.ndarray, sample_rate: int = 16000,
                            filter_confidence: bool = True, filter_fillers: bool = True,
                            min_confidence: float = 0.3) -> List[FasterTranscriptionSegment]:
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
        
        print(f"Initial segments: {len(segments)}")
        
        # Apply filters
        if filter_confidence:
            segments = self.filter_low_confidence_segments(segments, min_confidence)
            print(f"After confidence filter: {len(segments)}")
        
        if filter_fillers:
            segments = self.filter_filler_words(segments)
            print(f"After filler filter: {len(segments)}")
        
        return segments