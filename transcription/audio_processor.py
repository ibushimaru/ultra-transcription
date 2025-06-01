"""
Audio file processing module for loading and preprocessing audio files.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional
import noisereduce as nr
from pydub import AudioSegment
from pathlib import Path


class AudioProcessor:
    """Handle audio file loading and preprocessing."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for processing
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to target sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.mp3', '.m4a', '.aac']:
            # Use pydub for compressed formats
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
        else:
            # Use librosa for WAV and other formats
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        return audio_data, self.sample_rate
    
    def reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction to audio data.
        
        Args:
            audio_data: Input audio data
            sample_rate: Sample rate of audio
            
        Returns:
            Noise-reduced audio data
        """
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio_data, 
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.8
        )
        
        return reduced_noise
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Normalized audio data
        """
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        return audio_data
    
    def preprocess_audio(self, file_path: str, reduce_noise: bool = True) -> Tuple[np.ndarray, int]:
        """
        Complete audio preprocessing pipeline.
        
        Args:
            file_path: Path to audio file
            reduce_noise: Whether to apply noise reduction
            
        Returns:
            Tuple of (processed_audio_data, sample_rate)
        """
        # Load audio
        audio_data, sample_rate = self.load_audio(file_path)
        
        # Apply noise reduction if requested
        if reduce_noise:
            audio_data = self.reduce_noise(audio_data, sample_rate)
        
        # Normalize audio
        audio_data = self.normalize_audio(audio_data)
        
        return audio_data, sample_rate
    
    def get_audio_duration(self, file_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        audio_data, sample_rate = self.load_audio(file_path)
        duration = len(audio_data) / sample_rate
        return duration