"""
Optimized output formatter with multiple data structure variants.
Addresses redundancy and provides purpose-specific formats.
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class OptimizedSegment:
    """Optimized segment structure with minimal redundancy."""
    id: int
    start: float  # Start time in seconds (primary time format)
    end: float    # End time in seconds
    text: str
    confidence: float
    speaker: str
    speaker_confidence: Optional[float] = None

class OptimizedOutputFormatter:
    """
    Optimized output formatter with multiple data structure variants.
    
    Provides:
    - Compact format (minimal redundancy)
    - Standard format (human-readable)
    - Extended format (detailed analysis)
    - API format (machine-readable)
    """
    
    def __init__(self):
        self.format_variants = {
            'compact': 'Minimal data for storage/transmission',
            'standard': 'Human-readable with time formats',
            'extended': 'Detailed with word-level information',
            'api': 'Machine-readable with metadata'
        }
    
    def prepare_optimized_data(self, segments: List[Dict], 
                             variant: str = 'standard') -> Dict[str, Any]:
        """
        Prepare data in optimized format based on variant.
        
        Args:
            segments: Raw transcription segments
            variant: Output variant ('compact', 'standard', 'extended', 'api')
            
        Returns:
            Optimized data structure
        """
        if variant == 'compact':
            return self._prepare_compact_format(segments)
        elif variant == 'standard':
            return self._prepare_standard_format(segments)
        elif variant == 'extended':
            return self._prepare_extended_format(segments)
        elif variant == 'api':
            return self._prepare_api_format(segments)
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(self.format_variants.keys())}")
    
    def _prepare_compact_format(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Compact format with minimal redundancy.
        
        Features:
        - Only essential data
        - Single time format (seconds)
        - No calculated fields
        - Minimal metadata
        """
        optimized_segments = []
        
        for i, seg in enumerate(segments, 1):
            optimized_segments.append({
                'id': i,
                's': round(seg.get('start_seconds', seg.get('start_time', 0)), 3),  # start
                'e': round(seg.get('end_seconds', seg.get('end_time', 0)), 3),      # end
                't': seg.get('text', '').strip(),                                   # text
                'c': round(seg.get('confidence', 0), 3),                           # confidence
                'sp': seg.get('speaker_id', 'UNK')                                 # speaker (shortened)
            })
        
        return {
            'v': '1.0',  # version
            'segments': optimized_segments,
            'meta': {
                'count': len(optimized_segments),
                'generated': datetime.now().isoformat()[:19]  # No microseconds
            }
        }
    
    def _prepare_standard_format(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Standard format optimized for human readability.
        
        Features:
        - Human-readable field names
        - Multiple time formats for convenience
        - Essential calculated fields
        - Speaker statistics
        """
        optimized_segments = []
        total_duration = 0
        speaker_stats = {}
        
        for i, seg in enumerate(segments, 1):
            start_seconds = seg.get('start_seconds', seg.get('start_time', 0))
            end_seconds = seg.get('end_seconds', seg.get('end_time', 0))
            duration = end_seconds - start_seconds
            total_duration += duration
            
            speaker_id = seg.get('speaker_id', 'SPEAKER_UNKNOWN')
            
            # Update speaker statistics
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {'segments': 0, 'duration': 0}
            speaker_stats[speaker_id]['segments'] += 1
            speaker_stats[speaker_id]['duration'] += duration
            
            optimized_segments.append({
                'segment_id': i,
                'start_seconds': round(start_seconds, 3),
                'end_seconds': round(end_seconds, 3),
                'duration_seconds': round(duration, 3),
                'start_time': self._seconds_to_timestamp(start_seconds),
                'end_time': self._seconds_to_timestamp(end_seconds),
                'text': seg.get('text', '').strip(),
                'confidence': round(seg.get('confidence', 0), 3),
                'speaker_id': speaker_id
            })
        
        return {
            'format_version': '2.0',
            'format_type': 'standard',
            'segments': optimized_segments,
            'summary': {
                'total_segments': len(optimized_segments),
                'total_duration_seconds': round(total_duration, 3),
                'average_confidence': round(
                    sum(seg.get('confidence', 0) for seg in segments) / len(segments), 3
                ) if segments else 0,
                'speaker_count': len(speaker_stats),
                'speaker_stats': {
                    speaker: {
                        'segments': stats['segments'],
                        'duration_seconds': round(stats['duration'], 3),
                        'percentage': round(stats['duration'] / total_duration * 100, 1) if total_duration > 0 else 0
                    }
                    for speaker, stats in speaker_stats.items()
                }
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _prepare_extended_format(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        Extended format with detailed analysis.
        
        Features:
        - Word-level information
        - Quality metrics
        - Detailed speaker analysis
        - Statistical information
        """
        extended_segments = []
        word_count = 0
        confidence_scores = []
        speaking_rates = []
        
        for i, seg in enumerate(segments, 1):
            start_seconds = seg.get('start_seconds', seg.get('start_time', 0))
            end_seconds = seg.get('end_seconds', seg.get('end_time', 0))
            duration = end_seconds - start_seconds
            text = seg.get('text', '').strip()
            words = text.split()
            word_count += len(words)
            
            confidence = seg.get('confidence', 0)
            confidence_scores.append(confidence)
            
            # Calculate speaking rate (words per minute)
            speaking_rate = (len(words) / duration * 60) if duration > 0 else 0
            speaking_rates.append(speaking_rate)
            
            # Quality assessment
            quality_score = self._calculate_quality_score(confidence, len(words), duration)
            
            extended_segments.append({
                'segment_id': i,
                'timing': {
                    'start_seconds': round(start_seconds, 3),
                    'end_seconds': round(end_seconds, 3),
                    'duration_seconds': round(duration, 3),
                    'start_timestamp': self._seconds_to_timestamp(start_seconds),
                    'end_timestamp': self._seconds_to_timestamp(end_seconds)
                },
                'content': {
                    'text': text,
                    'word_count': len(words),
                    'character_count': len(text),
                    'words': words  # Individual words for word-level analysis
                },
                'quality': {
                    'confidence': round(confidence, 3),
                    'quality_score': round(quality_score, 3),
                    'speaking_rate_wpm': round(speaking_rate, 1)
                },
                'speaker': {
                    'id': seg.get('speaker_id', 'SPEAKER_UNKNOWN'),
                    'confidence': round(seg.get('speaker_confidence', 0), 3) if seg.get('speaker_confidence') is not None else None
                }
            })
        
        # Calculate advanced statistics
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        confidence_std = np.std(confidence_scores) if confidence_scores else 0
        avg_speaking_rate = np.mean(speaking_rates) if speaking_rates else 0
        
        return {
            'format_version': '2.0',
            'format_type': 'extended',
            'segments': extended_segments,
            'statistics': {
                'segment_count': len(extended_segments),
                'total_words': word_count,
                'total_characters': sum(len(seg.get('text', '')) for seg in segments),
                'confidence_statistics': {
                    'mean': round(avg_confidence, 3),
                    'std': round(confidence_std, 3),
                    'min': round(min(confidence_scores), 3) if confidence_scores else 0,
                    'max': round(max(confidence_scores), 3) if confidence_scores else 0
                },
                'speaking_rate_statistics': {
                    'mean_wpm': round(avg_speaking_rate, 1),
                    'std_wpm': round(np.std(speaking_rates), 1) if speaking_rates else 0
                },
                'quality_distribution': self._calculate_quality_distribution(extended_segments)
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _prepare_api_format(self, segments: List[Dict]) -> Dict[str, Any]:
        """
        API format optimized for machine processing.
        
        Features:
        - Consistent data types
        - Validation information
        - Processing metadata
        - Batch processing support
        """
        api_segments = []
        
        for i, seg in enumerate(segments, 1):
            api_segments.append({
                'id': i,
                'start_time': float(seg.get('start_seconds', seg.get('start_time', 0))),
                'end_time': float(seg.get('end_seconds', seg.get('end_time', 0))),
                'text': str(seg.get('text', '')).strip(),
                'confidence': float(seg.get('confidence', 0)),
                'speaker_id': str(seg.get('speaker_id', 'SPEAKER_UNKNOWN')),
                'speaker_confidence': float(seg.get('speaker_confidence', 0)) if seg.get('speaker_confidence') is not None else None,
                'metadata': {
                    'word_count': len(str(seg.get('text', '')).split()),
                    'duration': float(seg.get('end_seconds', seg.get('end_time', 0))) - float(seg.get('start_seconds', seg.get('start_time', 0)))
                }
            })
        
        return {
            'api_version': '1.0',
            'format': 'json',
            'data': {
                'segments': api_segments,
                'total_count': len(api_segments)
            },
            'validation': {
                'schema_version': '1.0',
                'required_fields': ['id', 'start_time', 'end_time', 'text', 'confidence', 'speaker_id'],
                'optional_fields': ['speaker_confidence', 'metadata']
            },
            'processing_info': {
                'generated_at': datetime.now().isoformat(),
                'format_type': 'api',
                'data_integrity': self._validate_data_integrity(api_segments)
            }
        }
    
    def _calculate_quality_score(self, confidence: float, word_count: int, duration: float) -> float:
        """Calculate overall quality score for a segment."""
        # Base score from confidence
        quality = confidence
        
        # Penalize very short segments (likely errors)
        if duration < 0.5:
            quality *= 0.7
        
        # Penalize segments with very few words relative to duration
        if duration > 0:
            words_per_second = word_count / duration
            if words_per_second < 0.5:  # Very slow speech
                quality *= 0.8
            elif words_per_second > 8:  # Very fast speech (likely error)
                quality *= 0.6
        
        return min(quality, 1.0)
    
    def _calculate_quality_distribution(self, segments: List[Dict]) -> Dict[str, int]:
        """Calculate quality distribution for extended format."""
        distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        for seg in segments:
            quality = seg['quality']['quality_score']
            if quality >= 0.9:
                distribution['excellent'] += 1
            elif quality >= 0.75:
                distribution['good'] += 1
            elif quality >= 0.6:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _validate_data_integrity(self, segments: List[Dict]) -> Dict[str, Any]:
        """Validate data integrity for API format."""
        issues = []
        
        for i, seg in enumerate(segments):
            # Check time consistency
            if seg['start_time'] >= seg['end_time']:
                issues.append(f"Segment {i+1}: Invalid time range")
            
            # Check confidence range
            if not 0 <= seg['confidence'] <= 1:
                issues.append(f"Segment {i+1}: Confidence out of range")
            
            # Check text content
            if not seg['text'].strip():
                issues.append(f"Segment {i+1}: Empty text")
        
        return {
            'is_valid': len(issues) == 0,
            'issue_count': len(issues),
            'issues': issues[:10]  # Limit to first 10 issues
        }
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS.mmm format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def save_optimized_format(self, segments: List[Dict], output_path: str, 
                            variant: str = 'standard', metadata: Optional[Dict] = None) -> str:
        """
        Save segments in optimized format.
        
        Args:
            segments: Transcription segments
            output_path: Output file path (without extension)
            variant: Format variant
            metadata: Additional metadata
            
        Returns:
            Path to saved file
        """
        data = self.prepare_optimized_data(segments, variant)
        
        # Add metadata if provided
        if metadata:
            if 'metadata' in data:
                data['metadata'].update(metadata)
            else:
                data['metadata'] = metadata
        
        # Determine file extension and format
        if variant == 'compact':
            file_path = f"{output_path}_compact.json"
        elif variant == 'extended':
            file_path = f"{output_path}_detailed.json"
        elif variant == 'api':
            file_path = f"{output_path}_api.json"
        else:
            file_path = f"{output_path}_optimized.json"
        
        # Save JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return file_path
    
    def save_optimized_csv(self, segments: List[Dict], output_path: str, 
                          variant: str = 'standard') -> str:
        """Save segments as optimized CSV."""
        data = self.prepare_optimized_data(segments, variant)
        csv_path = f"{output_path}_optimized.csv"
        
        if variant == 'compact':
            # Compact CSV format
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'start_sec', 'end_sec', 'text', 'confidence', 'speaker'])
                
                for seg in data['segments']:
                    writer.writerow([
                        seg['id'], seg['s'], seg['e'], 
                        seg['t'], seg['c'], seg['sp']
                    ])
        
        elif variant == 'extended':
            # Extended CSV format with quality metrics
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'segment_id', 'start_seconds', 'end_seconds', 'duration',
                    'text', 'word_count', 'confidence', 'quality_score',
                    'speaking_rate_wpm', 'speaker_id', 'speaker_confidence'
                ])
                
                for seg in data['segments']:
                    writer.writerow([
                        seg['segment_id'],
                        seg['timing']['start_seconds'],
                        seg['timing']['end_seconds'],
                        seg['timing']['duration_seconds'],
                        seg['content']['text'],
                        seg['content']['word_count'],
                        seg['quality']['confidence'],
                        seg['quality']['quality_score'],
                        seg['quality']['speaking_rate_wpm'],
                        seg['speaker']['id'],
                        seg['speaker']['confidence']
                    ])
        
        else:  # standard format
            # Standard optimized CSV (no time redundancy)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'segment_id', 'start_seconds', 'end_seconds', 'text', 
                    'confidence', 'speaker_id'
                ])
                
                for seg in data['segments']:
                    writer.writerow([
                        seg['segment_id'],
                        seg['start_seconds'],
                        seg['end_seconds'],
                        seg['text'],
                        seg['confidence'],
                        seg['speaker_id']
                    ])
        
        return csv_path
    
    def get_format_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison of different format variants."""
        return {
            'compact': {
                'description': 'Minimal storage, fastest processing',
                'use_cases': ['Storage optimization', 'Network transmission', 'Caching'],
                'features': ['Single time format', 'Shortened field names', 'No redundancy'],
                'size_reduction': '40-50%',
                'best_for': 'High-volume processing, mobile apps'
            },
            'standard': {
                'description': 'Human-readable, balanced approach',
                'use_cases': ['General use', 'Manual review', 'Analysis'],
                'features': ['Multiple time formats', 'Speaker statistics', 'Summary data'],
                'size_reduction': '10-20%',
                'best_for': 'Most common use cases'
            },
            'extended': {
                'description': 'Detailed analysis, research-oriented',
                'use_cases': ['Research', 'Quality analysis', 'Advanced processing'],
                'features': ['Word-level data', 'Quality metrics', 'Statistical analysis'],
                'size_reduction': '-20-30% (larger)',
                'best_for': 'Research, quality assessment'
            },
            'api': {
                'description': 'Machine processing, validation included',
                'use_cases': ['API responses', 'System integration', 'Validation'],
                'features': ['Type consistency', 'Data validation', 'Processing metadata'],
                'size_reduction': '0-10%',
                'best_for': 'System integration, APIs'
            }
        }