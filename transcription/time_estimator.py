"""
Processing time estimation for transcription tasks.
"""

import time
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    audio_duration: float
    processing_time: float
    model_size: str
    device: str
    engine: str
    enhanced_preprocessing: bool = False
    post_processing: bool = False
    speaker_diarization: bool = False


class TranscriptionTimeEstimator:
    """Estimate processing times for transcription tasks."""
    
    def __init__(self):
        """Initialize time estimator with performance benchmarks."""
        # Performance benchmarks (processing_time / audio_duration ratio)
        # Based on actual measurements
        self.benchmarks = {
            # Standard Whisper
            'whisper': {
                'tiny': {'cpu': 0.25, 'cuda': 0.15},
                'base': {'cpu': 0.45, 'cuda': 0.25},
                'small': {'cpu': 0.80, 'cuda': 0.45},
                'medium': {'cpu': 1.50, 'cuda': 0.80},
                'large': {'cpu': 3.00, 'cuda': 1.20}
            },
            # faster-whisper (more efficient)
            'faster-whisper': {
                'tiny': {'cpu': 0.15, 'cuda': 0.08},
                'base': {'cpu': 0.25, 'cuda': 0.12},
                'small': {'cpu': 0.45, 'cuda': 0.20},
                'medium': {'cpu': 0.80, 'cuda': 0.35},
                'large': {'cpu': 1.50, 'cuda': 0.60}
            },
            # ULTRA enhanced (calibrated with actual performance)
            'ultra-enhanced': {
                'tiny': {'cpu': 0.08, 'cuda': 0.04},
                'base': {'cpu': 0.14, 'cuda': 0.07},
                'small': {'cpu': 0.25, 'cuda': 0.12},
                'medium': {'cpu': 0.45, 'cuda': 0.20},
                'large': {'cpu': 0.80, 'cuda': 0.32}
            },
            # Chunked enhanced (memory-efficient for large files)
            'chunked-enhanced': {
                'tiny': {'cpu': 0.10, 'cuda': 0.05},
                'base': {'cpu': 0.18, 'cuda': 0.09},
                'small': {'cpu': 0.32, 'cuda': 0.15},
                'medium': {'cpu': 0.58, 'cuda': 0.25},
                'large': {'cpu': 1.00, 'cuda': 0.40}
            },
            # Maximum precision (ensemble + all enhancements)
            'maximum-precision': {
                'tiny': {'cpu': 0.25, 'cuda': 0.12},
                'base': {'cpu': 0.45, 'cuda': 0.20},
                'small': {'cpu': 0.80, 'cuda': 0.35},
                'medium': {'cpu': 1.40, 'cuda': 0.60},
                'large': {'cpu': 2.50, 'cuda': 1.00}
            },
            # Maximum precision chunked (for large files)
            'maximum-precision-chunked': {
                'tiny': {'cpu': 0.20, 'cuda': 0.10},
                'base': {'cpu': 0.35, 'cuda': 0.18},
                'small': {'cpu': 0.65, 'cuda': 0.30},
                'medium': {'cpu': 1.15, 'cuda': 0.50},
                'large': {'cpu': 2.00, 'cuda': 0.80}
            }
        }
        
        # Additional processing overhead factors (calibrated)
        self.overhead_factors = {
            'enhanced_preprocessing': 0.08,  # +8% for advanced audio processing (reduced)
            'post_processing': 0.05,         # +5% for text corrections (reduced)
            'speaker_diarization': 0.20,     # +20% for speaker identification (reduced)
            'vad': 0.03,                     # +3% for voice activity detection (reduced)
            'noise_reduction': 0.05          # +5% for noise reduction (reduced)
        }
        
        # Historical performance data for learning
        self.performance_history = []
    
    def estimate_processing_time(self, 
                                audio_duration: float,
                                model_size: str = 'base',
                                device: str = 'cpu',
                                engine: str = 'faster-whisper',
                                enhanced_preprocessing: bool = False,
                                post_processing: bool = False,
                                speaker_diarization: bool = False,
                                confidence_level: float = 0.8) -> Dict[str, float]:
        """
        Estimate processing time for transcription.
        
        Args:
            audio_duration: Duration of audio in seconds
            model_size: Whisper model size
            device: Processing device (cpu/cuda)
            engine: Engine type
            enhanced_preprocessing: Whether enhanced preprocessing is enabled
            post_processing: Whether post-processing is enabled
            speaker_diarization: Whether speaker diarization is enabled
            confidence_level: Confidence level for estimation (0.5-0.95)
            
        Returns:
            Dictionary with time estimates
        """
        # Get base processing ratio
        if engine in self.benchmarks and model_size in self.benchmarks[engine]:
            base_ratio = self.benchmarks[engine][model_size].get(device, 
                        self.benchmarks[engine][model_size]['cpu'])
        else:
            # Fallback to faster-whisper base
            base_ratio = self.benchmarks['faster-whisper']['base'][device]
        
        # Calculate base processing time
        base_time = audio_duration * base_ratio
        
        # Add overhead factors
        total_overhead = 0.0
        active_features = []
        
        if enhanced_preprocessing:
            total_overhead += self.overhead_factors['enhanced_preprocessing']
            active_features.append('Enhanced Preprocessing')
        
        if post_processing:
            total_overhead += self.overhead_factors['post_processing']
            active_features.append('Post-processing')
        
        if speaker_diarization:
            total_overhead += self.overhead_factors['speaker_diarization']
            active_features.append('Speaker Diarization')
        
        # Always add some base overhead
        total_overhead += self.overhead_factors['vad']  # VAD is usually enabled
        total_overhead += self.overhead_factors['noise_reduction']  # Usually enabled
        
        # Calculate estimated time
        estimated_time = base_time * (1 + total_overhead)
        
        # Add confidence interval
        confidence_multiplier = 1.0 + (1.0 - confidence_level) * 0.5
        
        # Calculate ranges
        optimistic = estimated_time * 0.7
        realistic = estimated_time * confidence_multiplier
        pessimistic = estimated_time * 1.5
        
        return {
            'optimistic': optimistic,
            'realistic': realistic,
            'pessimistic': pessimistic,
            'base_processing': base_time,
            'overhead_percentage': total_overhead * 100,
            'active_features': active_features,
            'processing_ratio': estimated_time / audio_duration
        }
    
    def format_time_estimate(self, seconds: float) -> str:
        """Format time estimate in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}時間"
    
    def display_time_estimate(self, audio_duration: float, estimates: Dict[str, float]):
        """Display formatted time estimation."""
        print("\n" + "⏱️" * 3 + " 処理時間予測 " + "⏱️" * 3)
        print("=" * 50)
        print(f"🎵 音声長: {self.format_time_estimate(audio_duration)}")
        print(f"⚡ 最短予想: {self.format_time_estimate(estimates['optimistic'])}")
        print(f"🎯 標準予想: {self.format_time_estimate(estimates['realistic'])}")
        print(f"⏳ 最長予想: {self.format_time_estimate(estimates['pessimistic'])}")
        print(f"📊 処理比率: {estimates['processing_ratio']:.2f}x")
        
        if estimates['active_features']:
            print(f"🔧 有効機能: {', '.join(estimates['active_features'])}")
        
        print(f"📈 オーバーヘッド: +{estimates['overhead_percentage']:.0f}%")
        print("=" * 50)
        
        # Provide contextual advice
        ratio = estimates['processing_ratio']
        if ratio < 0.3:
            print("🚀 高速処理が期待できます")
        elif ratio < 0.6:
            print("⚡ 標準的な処理速度です")
        elif ratio < 1.0:
            print("🐢 やや時間がかかります")
        else:
            print("⏰ 長時間の処理になります - コーヒーブレイクをどうぞ")
    
    def track_actual_performance(self, metrics: ProcessingMetrics):
        """Track actual performance for future estimation improvement."""
        self.performance_history.append(metrics)
        
        # Keep only recent 100 measurements
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_progress_callback(self, estimated_time: float):
        """Get a progress callback function for real-time updates."""
        start_time = time.time()
        
        def progress_callback(current_step: str, progress: float = None):
            """Progress callback function."""
            elapsed = time.time() - start_time
            
            if progress:
                remaining = (estimated_time - elapsed) * (1 - progress) / progress
                print(f"📊 {current_step}: {progress:.1%} - 残り約{self.format_time_estimate(remaining)}")
            else:
                remaining = estimated_time - elapsed
                print(f"🔧 {current_step} - 残り約{self.format_time_estimate(max(0, remaining))}")
        
        return progress_callback
    
    def estimate_for_batch(self, audio_files: list, **kwargs) -> Dict:
        """Estimate processing time for multiple audio files."""
        total_duration = 0
        individual_estimates = []
        
        for file_path in audio_files:
            # This would need audio duration detection
            # For now, assume average duration
            duration = 600  # 10 minutes default
            total_duration += duration
            
            estimate = self.estimate_processing_time(duration, **kwargs)
            individual_estimates.append({
                'file': file_path,
                'duration': duration,
                'estimate': estimate
            })
        
        # Total estimate with some batch efficiency
        batch_efficiency = 0.9  # 10% efficiency gain for batch processing
        total_estimate = self.estimate_processing_time(total_duration, **kwargs)
        
        for key in ['optimistic', 'realistic', 'pessimistic']:
            total_estimate[key] *= batch_efficiency
        
        return {
            'total_duration': total_duration,
            'total_estimate': total_estimate,
            'individual_estimates': individual_estimates,
            'batch_efficiency': batch_efficiency
        }
    
    def learn_from_performance(self):
        """Update benchmarks based on historical performance."""
        if len(self.performance_history) < 10:
            return
        
        # Group by configuration
        config_groups = {}
        
        for metrics in self.performance_history:
            config = f"{metrics.engine}_{metrics.model_size}_{metrics.device}"
            if config not in config_groups:
                config_groups[config] = []
            
            ratio = metrics.processing_time / metrics.audio_duration
            config_groups[config].append(ratio)
        
        # Update benchmarks with moving average
        for config, ratios in config_groups.items():
            if len(ratios) >= 5:  # Need enough samples
                avg_ratio = sum(ratios) / len(ratios)
                # Update benchmark (weighted average with existing)
                engine, model, device = config.split('_')
                if engine in self.benchmarks and model in self.benchmarks[engine]:
                    current = self.benchmarks[engine][model][device]
                    # 70% current + 30% new measurement
                    self.benchmarks[engine][model][device] = current * 0.7 + avg_ratio * 0.3