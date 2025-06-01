"""
Chunked transcription for large audio files (2+ hours).
Memory-efficient processing with automatic chunk management.
"""

import os
import math
import time
import tempfile
from typing import List, Dict, Optional, Generator, Tuple
from dataclasses import dataclass
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

from .faster_transcriber import FasterTranscriber
from .enhanced_audio_processor import EnhancedAudioProcessor


@dataclass
class ChunkInfo:
    """Information about an audio chunk."""
    index: int
    start_time: float
    end_time: float
    duration: float
    file_path: Optional[str] = None
    processed: bool = False


@dataclass 
class ChunkedSegment:
    """Transcription segment with global timing."""
    start_time: float
    end_time: float
    text: str
    confidence: float
    chunk_index: int
    original_start: float  # Original time within chunk


class ChunkedTranscriber:
    """Memory-efficient transcriber for large audio files using chunk processing."""
    
    def __init__(self, 
                 model_size: str = 'base',
                 language: str = 'ja',
                 device: str = 'cpu',
                 chunk_duration: int = 10,  # minutes
                 overlap_duration: float = 2.0,  # seconds
                 max_memory_mb: int = 512):
        """
        Initialize chunked transcriber.
        
        Args:
            model_size: Whisper model size
            language: Target language
            device: Processing device (cpu/cuda)
            chunk_duration: Duration of each chunk in minutes
            overlap_duration: Overlap between chunks in seconds
            max_memory_mb: Maximum memory usage per chunk
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.chunk_duration = chunk_duration * 60  # Convert to seconds
        self.overlap_duration = overlap_duration
        self.max_memory_mb = max_memory_mb
        
        # Initialize processors
        self.transcriber = FasterTranscriber(
            model_size=model_size,
            language=language, 
            device=device
        )
        self.audio_processor = EnhancedAudioProcessor()
        
        # Temporary directory for chunks
        self.temp_dir = tempfile.mkdtemp(prefix="chunked_transcription_")
        
    def estimate_optimal_chunk_size(self, audio_duration: float, file_size_mb: float) -> int:
        """
        Estimate optimal chunk size based on file characteristics.
        
        Args:
            audio_duration: Total audio duration in seconds
            file_size_mb: File size in MB
            
        Returns:
            Optimal chunk duration in seconds
        """
        # Base chunk size (10 minutes for most cases)
        base_chunk = 600  # 10 minutes
        
        # Adjust based on file size and available memory
        mb_per_second = file_size_mb / audio_duration
        memory_per_second = mb_per_second * 5  # Estimate processing overhead
        
        # Calculate max chunk size to stay under memory limit
        max_chunk_for_memory = (self.max_memory_mb * 0.8) / memory_per_second
        
        # Use smaller of base size or memory-constrained size
        optimal_chunk = min(base_chunk, max_chunk_for_memory)
        
        # Ensure minimum chunk size (2 minutes)
        optimal_chunk = max(120, optimal_chunk)
        
        print(f"📊 Memory analysis: {mb_per_second:.2f}MB/sec, optimal chunk: {optimal_chunk/60:.1f}min")
        return int(optimal_chunk)
    
    def create_chunks(self, audio_file: str) -> Tuple[List[ChunkInfo], float]:
        """
        Create audio chunks with overlap for seamless processing.
        
        Args:
            audio_file: Path to input audio file
            
        Returns:
            List of chunk information and total duration
        """
        print(f"🔪 Creating chunks from: {audio_file}")
        
        # Load audio metadata without loading full file
        try:
            audio = AudioSegment.from_file(audio_file)
            total_duration = len(audio) / 1000.0  # Convert to seconds
            
            # Get file size
            file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
            print(f"📏 File analysis: {total_duration/60:.1f}min, {file_size_mb:.1f}MB")
            
            # Optimize chunk size
            optimal_chunk_duration = self.estimate_optimal_chunk_size(total_duration, file_size_mb)
            
        except Exception as e:
            print(f"⚠️  Could not analyze file, using default chunk size: {e}")
            audio = AudioSegment.from_file(audio_file)
            total_duration = len(audio) / 1000.0
            optimal_chunk_duration = self.chunk_duration
        
        chunks = []
        chunk_index = 0
        start_time = 0
        
        while start_time < total_duration:
            # Calculate chunk boundaries
            end_time = min(start_time + optimal_chunk_duration, total_duration)
            chunk_duration = end_time - start_time
            
            # Create chunk info
            chunk_info = ChunkInfo(
                index=chunk_index,
                start_time=start_time,
                end_time=end_time,
                duration=chunk_duration
            )
            
            chunks.append(chunk_info)
            print(f"📦 Chunk {chunk_index}: {start_time/60:.1f}-{end_time/60:.1f}min ({chunk_duration/60:.1f}min)")
            
            # Move to next chunk with overlap consideration
            if end_time >= total_duration:
                break
                
            # Next chunk starts with overlap
            start_time = end_time - self.overlap_duration
            chunk_index += 1
        
        print(f"✅ Created {len(chunks)} chunks for {total_duration/60:.1f}min audio")
        return chunks, total_duration
    
    def extract_chunk(self, audio_file: str, chunk_info: ChunkInfo) -> str:
        """
        Extract a specific chunk from audio file.
        
        Args:
            audio_file: Source audio file
            chunk_info: Chunk information
            
        Returns:
            Path to extracted chunk file
        """
        try:
            # Load only the required segment
            audio = AudioSegment.from_file(audio_file)
            
            start_ms = int(chunk_info.start_time * 1000)
            end_ms = int(chunk_info.end_time * 1000)
            
            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]
            
            # Save chunk to temp file
            chunk_path = os.path.join(self.temp_dir, f"chunk_{chunk_info.index:03d}.wav")
            chunk_audio.export(chunk_path, format="wav")
            
            chunk_info.file_path = chunk_path
            return chunk_path
            
        except Exception as e:
            print(f"❌ Error extracting chunk {chunk_info.index}: {e}")
            raise
    
    def process_chunk(self, chunk_info: ChunkInfo, 
                     enhanced_preprocessing: bool = True,
                     filter_fillers: bool = True,
                     min_confidence: float = 0.3) -> List[ChunkedSegment]:
        """
        Process a single chunk with memory-efficient operations.
        
        Args:
            chunk_info: Chunk to process
            enhanced_preprocessing: Enable enhanced audio preprocessing
            filter_fillers: Filter out filler words
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of transcription segments with global timing
        """
        print(f"🔄 Processing chunk {chunk_info.index} ({chunk_info.duration/60:.1f}min)")
        
        try:
            # Memory-efficient audio processing
            if enhanced_preprocessing:
                # Process chunk with reduced memory footprint
                audio_data, sample_rate = self.audio_processor.advanced_preprocess_audio(
                    chunk_info.file_path,
                    enable_noise_reduction=True,
                    enable_speech_enhancement=True,
                    enable_spectral_norm=True,  # Enable but use memory-efficient mode
                    enable_volume_adjustment=True,
                    enable_silence_removal=False,
                    memory_efficient=True  # Use memory-efficient processing
                )
            else:
                audio_data, sample_rate = self.audio_processor.preprocess_audio(
                    chunk_info.file_path,
                    reduce_noise=True
                )
            
            # Transcribe chunk
            segments = self.transcriber.process_transcription(
                audio_data,
                sample_rate,
                filter_confidence=True,
                filter_fillers=filter_fillers,
                min_confidence=min_confidence
            )
            
            # Convert to chunked segments with global timing
            chunked_segments = []
            for seg in segments:
                # Adjust timing to global coordinates
                global_start = chunk_info.start_time + seg.start_time
                global_end = chunk_info.start_time + seg.end_time
                
                chunked_segment = ChunkedSegment(
                    start_time=global_start,
                    end_time=global_end,
                    text=seg.text,
                    confidence=seg.confidence,
                    chunk_index=chunk_info.index,
                    original_start=seg.start_time
                )
                
                chunked_segments.append(chunked_segment)
            
            # Clean up memory
            del audio_data
            
            chunk_info.processed = True
            print(f"✅ Chunk {chunk_info.index}: {len(chunked_segments)} segments")
            
            return chunked_segments
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_info.index}: {e}")
            return []
    
    def merge_overlapping_segments(self, all_segments: List[ChunkedSegment]) -> List[ChunkedSegment]:
        """
        Merge segments from overlapping chunks to avoid duplicates.
        
        Args:
            all_segments: All segments from all chunks
            
        Returns:
            Deduplicated and merged segments
        """
        if not all_segments:
            return []
        
        print("🔗 Merging overlapping segments...")
        
        # Sort by start time
        all_segments.sort(key=lambda x: x.start_time)
        
        merged_segments = []
        current_segment = all_segments[0]
        
        for next_segment in all_segments[1:]:
            # Check for overlap
            overlap_start = max(current_segment.start_time, next_segment.start_time)
            overlap_end = min(current_segment.end_time, next_segment.end_time)
            
            if overlap_start < overlap_end:
                # Segments overlap - choose the one with higher confidence
                if next_segment.confidence > current_segment.confidence:
                    current_segment = next_segment
                # Skip the lower confidence segment
            else:
                # No overlap - add current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        print(f"✅ Merged {len(all_segments)} → {len(merged_segments)} segments")
        return merged_segments
    
    def process_large_file(self, audio_file: str,
                          enhanced_preprocessing: bool = True,
                          filter_fillers: bool = True,
                          min_confidence: float = 0.3,
                          progress_callback = None) -> List[Dict]:
        """
        Process large audio file using chunked approach.
        
        Args:
            audio_file: Path to input audio file
            enhanced_preprocessing: Enable enhanced preprocessing
            filter_fillers: Filter filler words
            min_confidence: Minimum confidence threshold
            progress_callback: Optional progress callback function
            
        Returns:
            List of transcription segments compatible with OutputFormatter
        """
        try:
            start_time = time.time()
            
            # Create chunks
            chunks, total_duration = self.create_chunks(audio_file)
            
            if progress_callback:
                progress_callback("chunking", 1.0, len(chunks))
            
            # Process each chunk
            all_segments = []
            total_chunks = len(chunks)
            
            for i, chunk_info in enumerate(chunks):
                if progress_callback:
                    progress_callback("extracting", i / total_chunks)
                
                # Extract chunk
                chunk_path = self.extract_chunk(audio_file, chunk_info)
                
                if progress_callback:
                    progress_callback("processing", i / total_chunks)
                
                # Process chunk
                chunk_segments = self.process_chunk(
                    chunk_info,
                    enhanced_preprocessing=enhanced_preprocessing,
                    filter_fillers=filter_fillers,
                    min_confidence=min_confidence
                )
                
                all_segments.extend(chunk_segments)
                
                # Clean up chunk file to save disk space
                if chunk_info.file_path and os.path.exists(chunk_info.file_path):
                    os.remove(chunk_info.file_path)
                
                progress = (i + 1) / total_chunks
                if progress_callback:
                    progress_callback("transcription", progress, len(all_segments))
            
            # Merge overlapping segments
            if progress_callback:
                progress_callback("merging", 0.5)
                
            merged_segments = self.merge_overlapping_segments(all_segments)
            
            if progress_callback:
                progress_callback("merging", 1.0)
            
            # Convert to output format
            output_segments = []
            for seg in merged_segments:
                output_segments.append({
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'text': seg.text,
                    'confidence': seg.confidence,
                    'speaker_id': 'SPEAKER_UNKNOWN'  # Speaker diarization can be added later
                })
            
            processing_time = time.time() - start_time
            
            print(f"\n🎉 Chunked processing completed!")
            print(f"⏱️  Total time: {processing_time/60:.1f}min")
            print(f"📊 Processing ratio: {processing_time/total_duration:.2f}x")
            print(f"📝 Total segments: {len(output_segments)}")
            
            return output_segments
            
        except Exception as e:
            print(f"❌ Chunked processing failed: {e}")
            raise
        finally:
            # Clean up temp directory
            self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        try:
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(self.temp_dir)
                print("🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()