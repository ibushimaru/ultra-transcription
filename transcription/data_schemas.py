"""
Standardized data schemas and validation for audio transcription system.
Provides consistent data structures across all system components.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json
from datetime import datetime


class QualityLevel(Enum):
    """Quality assessment levels for transcription segments."""
    OUTSTANDING = "outstanding"  # 95%+
    EXCELLENT = "excellent"      # 90-95%
    VERY_GOOD = "very_good"      # 85-90%
    GOOD = "good"                # 80-85%
    FAIR = "fair"                # 70-80%
    POOR = "poor"                # <70%


class SpeakerConfidenceLevel(Enum):
    """Speaker identification confidence levels."""
    HIGH = "high"       # 85%+
    MEDIUM = "medium"   # 70-85%
    LOW = "low"         # 50-70%
    VERY_LOW = "very_low"  # <50%


class ProcessingEngine(Enum):
    """Available processing engines."""
    GPU_ULTRA_PRECISION = "gpu_ultra_precision"
    ULTRA_PRECISION = "ultra_precision"
    ENHANCED_TURBO = "enhanced_turbo"
    MAXIMUM_PRECISION = "maximum_precision"
    TURBO_REALTIME = "turbo_realtime"


@dataclass
class TimingInfo:
    """Standardized timing information."""
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    start_timestamp: str  # HH:MM:SS.mmm format
    end_timestamp: str


@dataclass
class ContentInfo:
    """Content analysis information."""
    text: str
    word_count: int
    character_count: int
    words: Optional[List[str]] = None
    language_detected: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    confidence: float
    quality_score: float
    quality_level: QualityLevel
    speaking_rate_wpm: Optional[float] = None
    noise_level: Optional[float] = None


@dataclass
class SpeakerInfo:
    """Speaker identification information."""
    id: str
    confidence: Optional[float] = None
    confidence_level: Optional[SpeakerConfidenceLevel] = None
    gender: Optional[str] = None
    estimated_age_range: Optional[str] = None


@dataclass
class ProcessingInfo:
    """Processing configuration and metadata."""
    engine: ProcessingEngine
    model_configuration: Dict[str, Any]
    device: str
    processing_time_seconds: float
    gpu_acceleration: bool = False
    techniques_applied: List[str] = None


@dataclass
class TranscriptionSegment:
    """
    Standardized transcription segment.
    Core data structure used across all system components.
    """
    segment_id: int
    timing: TimingInfo
    content: ContentInfo
    quality: QualityMetrics
    speaker: SpeakerInfo
    processing_metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper enum serialization."""
        data = asdict(self)
        
        # Convert enums to string values
        if isinstance(data['quality']['quality_level'], QualityLevel):
            data['quality']['quality_level'] = data['quality']['quality_level'].value
        
        if data['speaker']['confidence_level'] and isinstance(data['speaker']['confidence_level'], SpeakerConfidenceLevel):
            data['speaker']['confidence_level'] = data['speaker']['confidence_level'].value
            
        return data
    
    @classmethod
    def from_legacy_dict(cls, legacy_data: Dict[str, Any], segment_id: int) -> 'TranscriptionSegment':
        """Convert from legacy dictionary format to standardized format."""
        start_seconds = legacy_data.get('start_seconds', legacy_data.get('start_time', 0))
        end_seconds = legacy_data.get('end_seconds', legacy_data.get('end_time', 0))
        duration = end_seconds - start_seconds
        text = legacy_data.get('text', '')
        confidence = legacy_data.get('confidence', 0.0)
        
        # Create timing info
        timing = TimingInfo(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=duration,
            start_timestamp=_seconds_to_timestamp(start_seconds),
            end_timestamp=_seconds_to_timestamp(end_seconds)
        )
        
        # Create content info
        words = text.split() if text else []
        content = ContentInfo(
            text=text,
            word_count=len(words),
            character_count=len(text),
            words=words
        )
        
        # Create quality metrics
        quality_level = _determine_quality_level(confidence)
        speaking_rate = (len(words) / duration * 60) if duration > 0 else 0
        quality = QualityMetrics(
            confidence=confidence,
            quality_score=confidence,  # Simplified for legacy
            quality_level=quality_level,
            speaking_rate_wpm=speaking_rate
        )
        
        # Create speaker info
        speaker_id = legacy_data.get('speaker_id', 'SPEAKER_UNKNOWN')
        speaker_confidence = legacy_data.get('speaker_confidence', 0.0)
        speaker_conf_level = _determine_speaker_confidence_level(speaker_confidence) if speaker_confidence else None
        
        speaker = SpeakerInfo(
            id=speaker_id,
            confidence=speaker_confidence,
            confidence_level=speaker_conf_level
        )
        
        return cls(
            segment_id=segment_id,
            timing=timing,
            content=content,
            quality=quality,
            speaker=speaker
        )


@dataclass
class SpeakerStatistics:
    """Speaker statistics for analysis."""
    speaker_id: str
    segment_count: int
    total_duration_seconds: float
    total_words: int
    average_confidence: float
    speaking_time_percentage: float
    estimated_characteristics: Optional[Dict[str, Any]] = None


@dataclass
class TranscriptionSummary:
    """Summary statistics for entire transcription."""
    total_segments: int
    total_duration_seconds: float
    total_words: int
    total_characters: int
    average_confidence: float
    quality_distribution: Dict[str, int]
    speaker_count: int
    speaker_statistics: List[SpeakerStatistics]
    processing_info: ProcessingInfo


@dataclass
class TranscriptionResult:
    """
    Complete transcription result with metadata.
    Top-level container for all transcription data.
    """
    format_version: str = "2.0"
    format_type: str = "standard"
    generated_at: str = None
    input_file: str = ""
    segments: List[TranscriptionSegment] = None
    summary: Optional[TranscriptionSummary] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now().isoformat()
        if self.segments is None:
            self.segments = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        
        # Convert segments
        data['segments'] = [seg.to_dict() if hasattr(seg, 'to_dict') else seg for seg in self.segments]
        
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def save_json(self, file_path: str) -> None:
        """Save to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


# Validation functions
def validate_transcription_result(result: TranscriptionResult) -> List[str]:
    """Validate transcription result and return list of issues."""
    issues = []
    
    if not result.segments:
        issues.append("No segments found")
        return issues
    
    for i, segment in enumerate(result.segments):
        # Check timing consistency
        if hasattr(segment, 'timing'):
            timing = segment.timing
            if timing.start_seconds >= timing.end_seconds:
                issues.append(f"Segment {i+1}: Invalid time range")
            
            if timing.duration_seconds != (timing.end_seconds - timing.start_seconds):
                issues.append(f"Segment {i+1}: Duration mismatch")
        
        # Check quality metrics
        if hasattr(segment, 'quality'):
            quality = segment.quality
            if not 0 <= quality.confidence <= 1:
                issues.append(f"Segment {i+1}: Confidence out of range")
        
        # Check content
        if hasattr(segment, 'content'):
            content = segment.content
            if not content.text.strip():
                issues.append(f"Segment {i+1}: Empty text")
            
            if content.word_count != len(content.text.split()):
                issues.append(f"Segment {i+1}: Word count mismatch")
    
    return issues


# Helper functions
def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _determine_quality_level(confidence: float) -> QualityLevel:
    """Determine quality level from confidence score."""
    if confidence >= 0.95:
        return QualityLevel.OUTSTANDING
    elif confidence >= 0.90:
        return QualityLevel.EXCELLENT
    elif confidence >= 0.85:
        return QualityLevel.VERY_GOOD
    elif confidence >= 0.80:
        return QualityLevel.GOOD
    elif confidence >= 0.70:
        return QualityLevel.FAIR
    else:
        return QualityLevel.POOR


def _determine_speaker_confidence_level(confidence: Optional[float]) -> Optional[SpeakerConfidenceLevel]:
    """Determine speaker confidence level from confidence score."""
    if confidence is None:
        return None
    
    if confidence >= 0.85:
        return SpeakerConfidenceLevel.HIGH
    elif confidence >= 0.70:
        return SpeakerConfidenceLevel.MEDIUM
    elif confidence >= 0.50:
        return SpeakerConfidenceLevel.LOW
    else:
        return SpeakerConfidenceLevel.VERY_LOW


# Format conversion utilities
def convert_legacy_to_standard(legacy_segments: List[Dict[str, Any]]) -> List[TranscriptionSegment]:
    """Convert legacy format segments to standardized format."""
    return [
        TranscriptionSegment.from_legacy_dict(seg, i+1) 
        for i, seg in enumerate(legacy_segments)
    ]


def export_to_api_format(result: TranscriptionResult) -> Dict[str, Any]:
    """Export to API-friendly format."""
    return {
        "api_version": "1.0",
        "format": "json",
        "data": result.to_dict(),
        "validation": {
            "schema_version": "2.0",
            "is_valid": len(validate_transcription_result(result)) == 0,
            "issues": validate_transcription_result(result)
        },
        "processing_info": {
            "generated_at": result.generated_at,
            "format_type": result.format_type
        }
    }