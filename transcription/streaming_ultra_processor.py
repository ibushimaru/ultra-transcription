#!/usr/bin/env python3
"""
Streaming Ultra Processor for Ultra Audio Transcription

Specialized for ultra-large files (2+ hours) with minimal memory usage:
- Real-time streaming processing
- Intelligent memory management
- Auto-adaptive chunk sizing based on available resources
- GPU memory optimization
- Continuous progress tracking with ETA
- Live quality monitoring
"""

import os
import sys
import json
import time
# import psutil  # Optional dependency
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import tempfile
import gc
from collections import deque
import threading
from queue import Queue, Empty
import signal

import librosa
import numpy as np
import torch

# Import our enhanced components
from .enhanced_audio_processor import EnhancedAudioProcessor
from .enhanced_speaker_diarization import EnhancedSpeakerDiarizer
from .transcriber import Transcriber
from .post_processor import TranscriptionPostProcessor
from .optimized_output_formatter import OptimizedOutputFormatter

@dataclass
class StreamingMetrics:
    """Real-time streaming metrics"""
    processed_duration: float = 0.0
    total_duration: float = 0.0
    current_chunk: int = 0
    total_chunks: int = 0
    segments_count: int = 0
    average_confidence: float = 0.0
    processing_speed: float = 0.0  # Real-time factor
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    eta_seconds: float = 0.0
    start_time: float = 0.0

class MemoryManager:
    """Intelligent memory management for streaming processing"""
    
    def __init__(self, target_memory_mb: float = 4000):
        self.target_memory_mb = target_memory_mb
        self.peak_memory_mb = 0.0
        
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current RAM and GPU memory usage in MB"""
        # RAM usage - simplified approach
        try:
            import resource
            ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB to MB
        except:
            ram_mb = 0.0  # Fallback
        
        # GPU memory usage
        gpu_mb = 0.0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        self.peak_memory_mb = max(self.peak_memory_mb, ram_mb + gpu_mb)
        return ram_mb, gpu_mb
    
    def optimize_memory(self):
        """Optimize memory usage"""
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def calculate_optimal_chunk_size(self, base_chunk_minutes: float = 10.0) -> float:
        """Calculate optimal chunk size based on available memory"""
        try:
            # Estimate available memory (simplified)
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                available_kb = 0
                for line in meminfo.split('\n'):
                    if 'MemAvailable:' in line:
                        available_kb = int(line.split()[1])
                        break
                available_memory = available_kb / 1024  # KB to MB
        except:
            # Fallback to conservative estimate
            available_memory = 4000  # 4GB default
        
        if available_memory > 8000:  # 8GB+
            return min(base_chunk_minutes * 1.5, 20.0)
        elif available_memory > 4000:  # 4GB+
            return base_chunk_minutes
        else:  # Low memory
            return max(base_chunk_minutes * 0.5, 5.0)

class StreamingUltraProcessor:
    """
    Streaming processor optimized for ultra-large audio files
    """
    
    def __init__(self,
                 base_chunk_minutes: float = 10.0,
                 overlap_seconds: float = 15.0,
                 max_memory_mb: float = 4000,
                 enable_gpu: bool = True,
                 quality_threshold: float = 0.7):
        """
        Initialize streaming processor
        
        Args:
            base_chunk_minutes: Base chunk size in minutes
            overlap_seconds: Overlap between chunks
            max_memory_mb: Maximum memory usage target
            enable_gpu: Whether to use GPU acceleration
            quality_threshold: Minimum quality threshold for segments
        """
        self.base_chunk_minutes = base_chunk_minutes
        self.overlap_seconds = overlap_seconds
        self.quality_threshold = quality_threshold
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(max_memory_mb)
        
        # Initialize components
        self.audio_processor = EnhancedAudioProcessor()
        self.speaker_diarizer = EnhancedSpeakerDiarizer(method="acoustic")  # Fast method
        self.output_formatter = OptimizedOutputFormatter()
        self.post_processor = TranscriptionPostProcessor()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Device detection with memory consideration
        self.device = self._detect_optimal_device(enable_gpu)
        
        # Streaming state
        self.is_processing = False
        self.should_stop = False
        
        # Metrics
        self.metrics = StreamingMetrics()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received shutdown signal, stopping processing...")
        self.should_stop = True
    
    def _detect_optimal_device(self, enable_gpu: bool) -> str:
        """Detect optimal device considering memory constraints"""
        if not enable_gpu:
            return "cpu"
        
        try:
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                if gpu_memory >= 6000:  # 6GB+ VRAM
                    # Test GPU functionality
                    test_tensor = torch.randn(100, 100).cuda()
                    torch.matmul(test_tensor, test_tensor)
                    del test_tensor
                    torch.cuda.empty_cache()
                    self.logger.info(f"GPU detected: {gpu_memory:.0f}MB VRAM")
                    return "cuda"
                else:
                    self.logger.info(f"GPU has insufficient VRAM: {gpu_memory:.0f}MB, using CPU")
        except Exception as e:
            self.logger.warning(f"GPU test failed: {e}, using CPU")
        
        return "cpu"
    
    def _create_streaming_chunks(self, audio_file: str) -> Iterator[Tuple[np.ndarray, int, float, float]]:
        """Create streaming audio chunks with adaptive sizing"""
        try:
            # Get audio metadata
            total_duration = librosa.get_duration(path=audio_file)
            self.metrics.total_duration = total_duration
            
            # Calculate optimal chunk size
            chunk_size = self.memory_manager.calculate_optimal_chunk_size(self.base_chunk_minutes) * 60.0
            
            # Calculate total chunks
            self.metrics.total_chunks = int(np.ceil(total_duration / chunk_size))
            
            self.logger.info(f"Streaming {total_duration:.1f}s audio in {self.metrics.total_chunks} chunks "
                           f"of {chunk_size/60.0:.1f} minutes each")
            
            chunk_id = 0
            start_time = 0.0
            
            while start_time < total_duration and not self.should_stop:
                # Calculate chunk boundaries
                end_time = min(start_time + chunk_size, total_duration)
                actual_end_time = min(end_time + self.overlap_seconds, total_duration)
                
                # Load chunk with memory optimization
                self.memory_manager.optimize_memory()
                
                try:
                    audio_data, sample_rate = librosa.load(
                        audio_file,
                        sr=22050,  # Lower sample rate for memory efficiency
                        offset=start_time,
                        duration=actual_end_time - start_time,
                        dtype=np.float32
                    )
                    
                    self.metrics.current_chunk = chunk_id + 1
                    
                    yield audio_data, sample_rate, start_time, end_time
                    
                    chunk_id += 1
                    start_time = end_time
                    
                except Exception as e:
                    self.logger.error(f"Failed to load chunk {chunk_id}: {e}")
                    start_time = end_time  # Skip this chunk
                    continue
                
        except Exception as e:
            self.logger.error(f"Failed to create streaming chunks: {e}")
            raise
    
    def _process_chunk_stream(self, audio_data: np.ndarray, sample_rate: int,
                             start_time: float, end_time: float,
                             transcriber, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single streaming chunk"""
        try:
            # Enhanced audio preprocessing
            processed_audio, processed_sr = self.audio_processor.advanced_preprocess_audio_data(
                audio_data, sample_rate,
                enable_noise_reduction=True,
                enable_speech_enhancement=True
            )
            
            # Transcribe chunk
            segment_results = transcriber.transcribe_audio(processed_audio, processed_sr)
            # Convert to expected format
            segments = [
                {
                    'start': seg.start_time,
                    'end': seg.end_time,
                    'text': seg.text,
                    'confidence': seg.confidence
                }
                for seg in segment_results
            ]
            
            # Apply Japanese post-processing if needed
            if settings.get('language') == 'ja':
                segments = self.post_processor.process_transcription(segments)
            
            # Adjust timestamps to global timeline
            for segment in segments:
                segment['start'] += start_time
                segment['end'] += start_time
                
                # Clip to chunk boundaries (remove overlap)
                if segment['end'] > end_time:
                    segment['end'] = end_time
                if segment['start'] >= end_time:
                    continue  # Skip segments outside chunk boundary
            
            # Filter by quality threshold
            quality_segments = [seg for seg in segments 
                              if seg.get('confidence', 0.0) >= self.quality_threshold]
            
            # Speaker diarization for quality segments
            if settings.get('enable_speaker_recognition', True) and quality_segments:
                try:
                    speaker_segments = self.speaker_diarizer.diarize_audio(
                        processed_audio, processed_sr,
                        method='acoustic',  # Fast method for streaming
                        num_speakers=settings.get('num_speakers')
                    )
                    
                    # Merge speaker information
                    quality_segments = self._merge_speaker_info_stream(
                        quality_segments, speaker_segments, start_time
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Speaker diarization failed: {e}")
                    for segment in quality_segments:
                        segment['speaker'] = 'SPEAKER_UNKNOWN'
            
            return quality_segments
            
        except Exception as e:
            self.logger.error(f"Chunk processing failed: {e}")
            return []
    
    def _merge_speaker_info_stream(self, transcription_segments: List[Dict],
                                  speaker_segments: List[Dict],
                                  chunk_start_time: float) -> List[Dict]:
        """Fast speaker info merging for streaming"""
        for t_seg in transcription_segments:
            t_start_local = t_seg['start'] - chunk_start_time
            t_end_local = t_seg['end'] - chunk_start_time
            
            # Find best matching speaker segment
            best_speaker = 'SPEAKER_UNKNOWN'
            max_overlap = 0.0
            
            for s_seg in speaker_segments:
                overlap_start = max(t_start_local, s_seg['start'])
                overlap_end = min(t_end_local, s_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = s_seg.get('speaker', 'SPEAKER_UNKNOWN')
            
            t_seg['speaker'] = best_speaker
        
        return transcription_segments
    
    def _update_metrics(self, segments: List[Dict], processing_time: float):
        """Update streaming metrics"""
        if segments:
            self.metrics.segments_count += len(segments)
            
            # Update confidence
            confidences = [seg.get('confidence', 0.0) for seg in segments]
            if confidences:
                new_avg = np.mean(confidences)
                if self.metrics.average_confidence == 0.0:
                    self.metrics.average_confidence = new_avg
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.metrics.average_confidence = (alpha * new_avg + 
                                                     (1 - alpha) * self.metrics.average_confidence)
        
        # Update processing speed (real-time factor)
        chunk_duration = segments[-1]['end'] - segments[0]['start'] if segments else 0.0
        if processing_time > 0 and chunk_duration > 0:
            speed = chunk_duration / processing_time
            if self.metrics.processing_speed == 0.0:
                self.metrics.processing_speed = speed
            else:
                alpha = 0.2
                self.metrics.processing_speed = (alpha * speed + 
                                               (1 - alpha) * self.metrics.processing_speed)
        
        # Update memory usage
        ram_mb, gpu_mb = self.memory_manager.get_memory_usage()
        self.metrics.memory_usage_mb = ram_mb
        self.metrics.gpu_memory_mb = gpu_mb
        
        # Update processed duration
        if segments:
            self.metrics.processed_duration = segments[-1]['end']
        
        # Calculate ETA
        remaining_duration = self.metrics.total_duration - self.metrics.processed_duration
        if self.metrics.processing_speed > 0:
            self.metrics.eta_seconds = remaining_duration / self.metrics.processing_speed
    
    def _log_progress(self):
        """Log current progress"""
        progress = (self.metrics.processed_duration / self.metrics.total_duration * 100) if self.metrics.total_duration > 0 else 0
        
        self.logger.info(
            f"Progress: {progress:.1f}% | "
            f"Chunk: {self.metrics.current_chunk}/{self.metrics.total_chunks} | "
            f"Segments: {self.metrics.segments_count} | "
            f"Confidence: {self.metrics.average_confidence:.3f} | "
            f"Speed: {self.metrics.processing_speed:.1f}x | "
            f"Memory: {self.metrics.memory_usage_mb:.0f}MB | "
            f"ETA: {self.metrics.eta_seconds/60:.1f}min"
        )
    
    def process_streaming(self, audio_file: str,
                         output_file: Optional[str] = None,
                         model: str = 'large-v3-turbo',
                         language: Optional[str] = None,
                         speaker_method: str = 'acoustic',
                         num_speakers: Optional[int] = None,
                         output_format: str = 'extended') -> Dict[str, Any]:
        """
        Process audio file with streaming ultra processor
        
        Args:
            audio_file: Path to audio file
            output_file: Output file base path
            model: Whisper model to use
            language: Language code
            speaker_method: Speaker diarization method
            num_speakers: Expected number of speakers
            output_format: Output format
            
        Returns:
            Processing results
        """
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        self.is_processing = True
        self.should_stop = False
        self.metrics = StreamingMetrics()
        self.metrics.start_time = time.time()
        
        settings = {
            'model': model,
            'language': language,
            'speaker_method': speaker_method,
            'num_speakers': num_speakers,
            'enable_speaker_recognition': speaker_method != 'off'
        }
        
        try:
            self.logger.info(f"Starting streaming processing: {audio_file}")
            self.logger.info(f"Device: {self.device}, Model: {model}")
            
            # Load transcriber once
            transcriber = Transcriber(model_size=model, language=language or 'ja')
            
            all_segments = []
            output_buffer = []
            
            # Process streaming chunks
            for audio_data, sample_rate, start_time, end_time in self._create_streaming_chunks(audio_file):
                if self.should_stop:
                    self.logger.info("Processing stopped by user")
                    break
                
                chunk_start_time = time.time()
                
                # Process chunk
                segments = self._process_chunk_stream(
                    audio_data, sample_rate, start_time, end_time,
                    transcriber, settings
                )
                
                processing_time = time.time() - chunk_start_time
                
                # Update metrics
                self._update_metrics(segments, processing_time)
                
                # Add to results
                all_segments.extend(segments)
                
                # Log progress
                self._log_progress()
                
                # Write incremental output if requested
                if output_file and segments:
                    output_buffer.extend(segments)
                    
                    # Write every 10 chunks or at end
                    if len(output_buffer) >= 50 or end_time >= self.metrics.total_duration:
                        self._write_incremental_output(output_buffer, output_file, output_format)
                        output_buffer = []
                
                # Memory optimization
                self.memory_manager.optimize_memory()
                
                # Short pause to prevent system overload
                time.sleep(0.1)
            
            # Final processing
            total_time = time.time() - self.metrics.start_time
            
            # Apply speaker consistency to final segments
            if settings['enable_speaker_recognition'] and all_segments:
                self.logger.info("Applying final speaker consistency...")
                try:
                    all_segments = self._apply_streaming_speaker_consistency(all_segments)
                except Exception as e:
                    self.logger.warning(f"Speaker consistency failed: {e}")
            
            # Create final result
            result = {
                'format_version': '2.0',
                'segments': all_segments,
                'metadata': {
                    'original_file': audio_file,
                    'total_duration': self.metrics.total_duration,
                    'total_segments': len(all_segments),
                    'processing_time': total_time,
                    'average_confidence': self.metrics.average_confidence,
                    'real_time_factor': self.metrics.processing_speed,
                    'peak_memory_mb': self.memory_manager.peak_memory_mb,
                    'device_used': self.device,
                    'processing_method': 'streaming_ultra',
                    'settings': settings
                }
            }
            
            # Write final output
            if output_file:
                self._write_final_output(result, output_file, output_format)
            
            self.logger.info(f"Streaming processing completed: {len(all_segments)} segments, "
                           f"avg confidence: {self.metrics.average_confidence:.3f}, "
                           f"speed: {self.metrics.processing_speed:.1f}x realtime")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Streaming processing failed: {e}")
            raise
        finally:
            self.is_processing = False
            self.memory_manager.optimize_memory()
    
    def _apply_streaming_speaker_consistency(self, segments: List[Dict]) -> List[Dict]:
        """Apply speaker consistency optimized for streaming results"""
        if len(segments) < 2:
            return segments
        
        # Simple consistency algorithm for streaming
        for i in range(1, len(segments)):
            current = segments[i]
            previous = segments[i-1]
            
            # If very short segment with same speaker, merge
            if (current['end'] - current['start'] < 2.0 and 
                current['speaker'] == previous['speaker'] and
                current['start'] - previous['end'] < 1.0):
                
                # Merge with previous
                previous['end'] = current['end']
                previous['text'] += ' ' + current['text']
                previous['confidence'] = (previous['confidence'] + current['confidence']) / 2
                segments[i] = None  # Mark for removal
        
        # Remove merged segments
        segments = [seg for seg in segments if seg is not None]
        
        return segments
    
    def _write_incremental_output(self, segments: List[Dict], output_file: str, output_format: str):
        """Write incremental output during processing"""
        try:
            base_path = Path(output_file).with_suffix('')
            temp_file = f"{base_path}_temp.jsonl"
            
            # Append segments to JSONL file
            with open(temp_file, 'a', encoding='utf-8') as f:
                for segment in segments:
                    f.write(json.dumps(segment, ensure_ascii=False) + '\n')
                    
        except Exception as e:
            self.logger.warning(f"Failed to write incremental output: {e}")
    
    def _write_final_output(self, result: Dict[str, Any], output_file: str, output_format: str):
        """Write final output files"""
        try:
            base_path = Path(output_file).with_suffix('')
            
            # JSON output
            json_file = f"{base_path}_streaming.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Optimized CSV output
            optimized_data = self.output_formatter.prepare_optimized_data(
                result['segments'], variant=output_format
            )
            csv_file = f"{base_path}_streaming.csv"
            self.output_formatter.save_csv(optimized_data, csv_file)
            
            # Clean up temp files
            temp_file = f"{base_path}_temp.jsonl"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            self.logger.info(f"Final output saved: {json_file}, {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to write final output: {e}")

def main():
    """Command-line interface for streaming ultra processor"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Streaming Ultra Processor for Large Audio Files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 2-hour audio with streaming
  python -m transcription.streaming_ultra_processor input.mp3 -o output
  
  # Low memory mode
  python -m transcription.streaming_ultra_processor input.mp3 -o output --max-memory 2000
  
  # CPU-only processing
  python -m transcription.streaming_ultra_processor input.mp3 -o output --device cpu
        """
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('-o', '--output', required=True, help='Output file base path')
    parser.add_argument('--model', default='large-v3-turbo',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3-turbo', 'turbo'],
                       help='Whisper model size (default: large-v3-turbo for maximum speed)')
    parser.add_argument('--language', help='Language code')
    parser.add_argument('--speaker-method', default='acoustic',
                       choices=['acoustic', 'clustering', 'off'],
                       help='Speaker diarization method')
    parser.add_argument('--num-speakers', type=int, help='Expected number of speakers')
    parser.add_argument('--chunk-size', type=float, default=10.0,
                       help='Base chunk size in minutes')
    parser.add_argument('--overlap', type=float, default=15.0,
                       help='Overlap in seconds')
    parser.add_argument('--max-memory', type=float, default=4000,
                       help='Maximum memory usage in MB')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Processing device')
    parser.add_argument('--output-format', default='extended',
                       choices=['compact', 'standard', 'extended', 'api'],
                       help='Output format')
    parser.add_argument('--quality-threshold', type=float, default=0.5,
                       help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = StreamingUltraProcessor(
        base_chunk_minutes=args.chunk_size,
        overlap_seconds=args.overlap,
        max_memory_mb=args.max_memory,
        enable_gpu=(args.device != 'cpu'),
        quality_threshold=args.quality_threshold
    )
    
    try:
        # Process file
        result = processor.process_streaming(
            audio_file=args.audio_file,
            output_file=args.output,
            model=args.model,
            language=args.language,
            speaker_method=args.speaker_method,
            num_speakers=args.num_speakers,
            output_format=args.output_format
        )
        
        print(f"\n✅ Streaming processing completed!")
        print(f"📊 Results:")
        print(f"   - Total segments: {result['metadata']['total_segments']}")
        print(f"   - Average confidence: {result['metadata']['average_confidence']:.1%}")
        print(f"   - Real-time factor: {result['metadata']['real_time_factor']:.1f}x")
        print(f"   - Peak memory: {result['metadata']['peak_memory_mb']:.0f}MB")
        print(f"   - Processing time: {result['metadata']['processing_time']:.1f}s")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()