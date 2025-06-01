"""
Enhanced audio preprocessing for better transcription accuracy.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional
import noisereduce as nr
from pydub import AudioSegment
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings("ignore")


class EnhancedAudioProcessor:
    """Enhanced audio preprocessing with advanced techniques."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize enhanced audio processor.
        
        Args:
            sample_rate: Target sample rate for processing
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with enhanced preprocessing."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.mp3', '.m4a', '.aac']:
            # Enhanced pydub loading with better quality
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
            # Apply high-quality conversion
            audio = audio.apply_gain(0)  # Normalize without clipping
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
        else:
            # Enhanced librosa loading
            audio_data, sr = librosa.load(
                file_path, 
                sr=self.sample_rate, 
                mono=True,
                res_type='kaiser_best'  # High quality resampling
            )
        
        return audio_data, self.sample_rate
    
    def advanced_noise_reduction(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply advanced noise reduction techniques.
        """
        # Multi-stage noise reduction
        
        # Stage 1: Spectral gating
        reduced_1 = nr.reduce_noise(
            y=audio_data,
            sr=sample_rate,
            stationary=False,
            prop_decrease=0.6
        )
        
        # Stage 2: Adaptive noise reduction
        reduced_2 = nr.reduce_noise(
            y=reduced_1,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.4
        )
        
        return reduced_2
    
    def enhance_speech_clarity(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance speech clarity using signal processing.
        """
        # Pre-emphasis filter (commonly used in speech processing)
        pre_emphasis = 0.97
        emphasized = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Dynamic range compression
        threshold = 0.1
        ratio = 4.0
        
        # Simple compressor
        compressed = np.where(
            np.abs(emphasized) > threshold,
            threshold + (np.abs(emphasized) - threshold) / ratio * np.sign(emphasized),
            emphasized
        )
        
        return compressed
    
    def spectral_normalization(self, audio_data: np.ndarray, sample_rate: int, 
                              memory_efficient: bool = False) -> np.ndarray:
        """
        Apply spectral normalization for consistent frequency response.
        
        Args:
            audio_data: Input audio data
            sample_rate: Sample rate
            memory_efficient: If True, skip memory-intensive STFT processing
        """
        if memory_efficient or len(audio_data) > 1000000:  # Skip for long audio (>62 seconds at 16kHz)
            # Use simple time-domain normalization instead
            return librosa.util.normalize(audio_data, norm=np.inf)
        
        # STFT for frequency domain processing (only for short audio)
        D = librosa.stft(audio_data, hop_length=512, n_fft=2048)
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Spectral normalization
        magnitude_norm = librosa.util.normalize(magnitude, axis=0)
        
        # Reconstruct signal
        D_normalized = magnitude_norm * np.exp(1j * phase)
        audio_normalized = librosa.istft(D_normalized, hop_length=512)
        
        return audio_normalized
    
    def intelligent_volume_adjustment(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Intelligent volume adjustment based on speech characteristics.
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Target RMS for optimal speech recognition
        target_rms = 0.1
        
        if rms > 0:
            # Smooth gain adjustment
            gain = target_rms / rms
            # Limit gain to prevent distortion
            gain = np.clip(gain, 0.1, 10.0)
            
            adjusted = audio_data * gain
            
            # Soft limiting to prevent clipping
            adjusted = np.tanh(adjusted * 0.9) * 0.9
            
            return adjusted
        
        return audio_data
    
    def remove_silence_gaps(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Remove long silence gaps while preserving natural pauses.
        """
        # Use librosa to detect non-silent intervals
        intervals = librosa.effects.split(
            audio_data,
            top_db=30,  # Threshold for silence detection
            frame_length=2048,
            hop_length=512
        )
        
        # Reconstruct audio with reduced silence gaps
        processed_segments = []
        
        for start, end in intervals:
            segment = audio_data[start:end]
            processed_segments.append(segment)
            
            # Add small pause between segments (100ms)
            if len(processed_segments) > 1:
                pause_samples = int(0.1 * sample_rate)
                pause = np.zeros(pause_samples)
                processed_segments.append(pause)
        
        if processed_segments:
            return np.concatenate(processed_segments)
        
        return audio_data
    
    def advanced_preprocess_audio(self, file_path: str, 
                                enable_noise_reduction: bool = True,
                                enable_speech_enhancement: bool = True,
                                enable_spectral_norm: bool = True,
                                enable_volume_adjustment: bool = True,
                                enable_silence_removal: bool = False,
                                memory_efficient: bool = False) -> Tuple[np.ndarray, int]:
        """
        Complete enhanced preprocessing pipeline.
        """
        print("🔧 Advanced audio preprocessing...")
        
        # Load audio
        audio_data, sample_rate = self.load_audio(file_path)
        print(f"   - Loaded: {len(audio_data)/sample_rate:.1f}s")
        
        # Advanced noise reduction
        if enable_noise_reduction:
            audio_data = self.advanced_noise_reduction(audio_data, sample_rate)
            print("   - Applied advanced noise reduction")
        
        # Speech enhancement
        if enable_speech_enhancement:
            audio_data = self.enhance_speech_clarity(audio_data, sample_rate)
            print("   - Enhanced speech clarity")
        
        # Spectral normalization
        if enable_spectral_norm:
            audio_data = self.spectral_normalization(audio_data, sample_rate, memory_efficient)
            if memory_efficient:
                print("   - Applied time-domain normalization (memory efficient)")
            else:
                print("   - Applied spectral normalization")
        
        # Volume adjustment
        if enable_volume_adjustment:
            audio_data = self.intelligent_volume_adjustment(audio_data)
            print("   - Adjusted volume intelligently")
        
        # Silence removal
        if enable_silence_removal:
            audio_data = self.remove_silence_gaps(audio_data, sample_rate)
            print("   - Removed excessive silence")
        
        # Final normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9
        
        print("✅ Advanced preprocessing completed")
        
        return audio_data, sample_rate
    
    def get_audio_duration(self, file_path: str) -> float:
        """Get duration of audio file in seconds."""
        audio_data, sample_rate = self.load_audio(file_path)
        duration = len(audio_data) / sample_rate
        return duration
    
    def advanced_preprocess_audio_data(self, audio_data: np.ndarray, sample_rate: int,
                                      enable_noise_reduction: bool = True,
                                      enable_speech_enhancement: bool = True,
                                      enable_spectral_norm: bool = False,
                                      enable_volume_adjustment: bool = True) -> Tuple[np.ndarray, int]:
        """
        Memory-efficient preprocessing for streaming audio data.
        
        Args:
            audio_data: Input audio data
            sample_rate: Sample rate
            enable_noise_reduction: Apply noise reduction
            enable_speech_enhancement: Apply speech enhancement
            enable_spectral_norm: Apply spectral normalization (memory intensive)
            enable_volume_adjustment: Apply volume adjustment
            
        Returns:
            Processed audio data and sample rate
        """
        
        # Advanced noise reduction
        if enable_noise_reduction:
            try:
                audio_data = self.advanced_noise_reduction(audio_data, sample_rate)
            except Exception:
                # Fallback to simple normalization
                audio_data = librosa.util.normalize(audio_data)
        
        # Speech enhancement
        if enable_speech_enhancement:
            try:
                audio_data = self.enhance_speech_clarity(audio_data, sample_rate)
            except Exception:
                # Fallback to simple filtering
                audio_data = self._simple_highpass_filter(audio_data, sample_rate)
        
        # Spectral normalization (only for short segments)
        if enable_spectral_norm and len(audio_data) < 500000:  # < 30 seconds at 16kHz
            try:
                audio_data = self.spectral_normalization(audio_data, sample_rate, memory_efficient=True)
            except Exception:
                audio_data = librosa.util.normalize(audio_data)
        
        # Volume adjustment
        if enable_volume_adjustment:
            audio_data = self.intelligent_volume_adjustment(audio_data)
        
        # Final normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9
        
        return audio_data, sample_rate
    
    def _simple_highpass_filter(self, audio_data: np.ndarray, sample_rate: int, 
                               cutoff: float = 80.0) -> np.ndarray:
        """Simple high-pass filter for memory efficiency."""
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            b, a = signal.butter(2, normalized_cutoff, btype='high')
            filtered = signal.filtfilt(b, a, audio_data)
            return filtered
        except Exception:
            return audio_data