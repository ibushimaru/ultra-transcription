"""
Enhanced speaker diarization with multiple fallback methods.
Improved speaker identification and data structure optimization.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import tempfile
import soundfile as sf
import os
import librosa
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import warnings

@dataclass
class EnhancedSpeakerSegment:
    """Enhanced data class for speaker segment with confidence."""
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None
    gender: Optional[str] = None  # 'male', 'female', 'unknown'
    
class EnhancedSpeakerDiarizer:
    """Enhanced speaker diarization with multiple methods and fallbacks."""
    
    def __init__(self, use_auth_token: Optional[str] = None, method: str = "auto"):
        """
        Initialize enhanced speaker diarizer.
        
        Args:
            use_auth_token: Hugging Face auth token for pyannote models
            method: Diarization method ('pyannote', 'acoustic', 'clustering', 'auto')
        """
        self.use_auth_token = use_auth_token
        self.method = method
        self.pyannote_pipeline = None
        self.available_methods = []
        self._initialize_methods()
    
    def _initialize_methods(self):
        """Initialize available diarization methods."""
        print("🎙️  話者識別システム初期化中...")
        
        # Try pyannote first (highest quality)
        if self._load_pyannote():
            self.available_methods.append("pyannote")
            print("✅ pyannote.audio 利用可能")
        
        # Always available fallback methods
        self.available_methods.extend(["acoustic", "clustering"])
        print(f"📊 利用可能な方法: {', '.join(self.available_methods)}")
        
        # Set method based on availability
        if self.method == "auto":
            self.method = self.available_methods[0]
        elif self.method not in self.available_methods:
            print(f"⚠️  指定された方法 '{self.method}' が利用できません。'{self.available_methods[0]}' を使用します。")
            self.method = self.available_methods[0]
        
        print(f"🔧 使用する話者識別方法: {self.method}")
    
    def _load_pyannote(self) -> bool:
        """Load pyannote diarization pipeline."""
        try:
            # Try to import pyannote first
            from pyannote.audio import Pipeline
            
            print("📥 pyannote speaker diarization model をロード中...")
            
            if self.use_auth_token:
                self.pyannote_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.use_auth_token
                )
                print("🔐 HuggingFace トークンで認証成功")
            else:
                # Try without auth token
                try:
                    self.pyannote_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                    print("🆓 トークンなしでpyannoteモデルをロード")
                except Exception as e:
                    print(f"⚠️  トークンなしでのpyannoteロードに失敗: {e}")
                    print("💡 HuggingFaceトークンを設定することで高精度な話者識別が可能になります")
                    return False
            
            # Set device
            if torch.cuda.is_available():
                self.pyannote_pipeline = self.pyannote_pipeline.to(torch.device("cuda"))
                print("🚀 GPU で pyannote を実行")
            else:
                print("💻 CPU で pyannote を実行")
            
            return True
            
        except ImportError:
            print("📦 pyannote.audio がインストールされていません")
            print("💡 pip install pyannote.audio でインストール可能")
            return False
        except Exception as e:
            print(f"❌ pyannote ロードエラー: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if speaker diarization is available."""
        return len(self.available_methods) > 0
    
    def diarize_audio(self, audio_data: np.ndarray, sample_rate: int, 
                     num_speakers: Optional[int] = None) -> List[EnhancedSpeakerSegment]:
        """
        Perform enhanced speaker diarization.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio data
            num_speakers: Expected number of speakers (optional)
            
        Returns:
            List of enhanced speaker segments
        """
        print(f"🎤 話者識別実行: {self.method} 方式")
        
        if self.method == "pyannote" and self.pyannote_pipeline:
            return self._diarize_with_pyannote(audio_data, sample_rate)
        elif self.method == "acoustic":
            return self._diarize_with_acoustic_features(audio_data, sample_rate, num_speakers)
        elif self.method == "clustering":
            return self._diarize_with_clustering(audio_data, sample_rate, num_speakers)
        else:
            print("⚠️  話者識別方法が利用できません。単一話者として処理します。")
            return self._create_single_speaker_segments(audio_data, sample_rate)
    
    def _diarize_with_pyannote(self, audio_data: np.ndarray, sample_rate: int) -> List[EnhancedSpeakerSegment]:
        """Pyannote.audio による高精度話者識別."""
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            sf.write(temp_path, audio_data, sample_rate)
        
        try:
            # Perform diarization
            diarization = self.pyannote_pipeline(temp_path)
            
            # Convert to enhanced speaker segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(EnhancedSpeakerSegment(
                    start_time=turn.start,
                    end_time=turn.end,
                    speaker_id=f"SPEAKER_{speaker.split('_')[-1].zfill(2)}",  # Format: SPEAKER_01
                    confidence=0.9  # pyannote は高信頼度
                ))
            
            print(f"✅ pyannote で {len(set(seg.speaker_id for seg in segments))} 話者を検出")
            return segments
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _diarize_with_acoustic_features(self, audio_data: np.ndarray, sample_rate: int, 
                                      num_speakers: Optional[int] = None) -> List[EnhancedSpeakerSegment]:
        """音響特徴量による話者識別."""
        print("🔊 音響特徴量による話者識別を実行中...")
        
        # Voice Activity Detection
        frame_length = int(0.025 * sample_rate)  # 25ms
        hop_length = int(0.010 * sample_rate)    # 10ms
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_data, 
            sr=sample_rate,
            n_mfcc=13,
            n_fft=frame_length,
            hop_length=hop_length
        )
        
        # Fundamental frequency (pitch) - important for speaker differentiation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            hop_length=hop_length
        )
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0)
        
        # Combine features
        features = np.vstack([mfcc, f0.reshape(1, -1), voiced_probs.reshape(1, -1)])
        
        # Segment audio into windows for clustering
        window_frames = int(2.0 * sample_rate / hop_length)  # 2秒ウィンドウ
        segments = []
        
        # Extract features for each window
        feature_vectors = []
        time_windows = []
        
        for i in range(0, features.shape[1] - window_frames, window_frames // 2):
            window_features = features[:, i:i+window_frames]
            feature_vector = np.mean(window_features, axis=1)  # Average over time
            feature_vectors.append(feature_vector)
            
            start_time = i * hop_length / sample_rate
            end_time = (i + window_frames) * hop_length / sample_rate
            time_windows.append((start_time, end_time))
        
        if not feature_vectors:
            return self._create_single_speaker_segments(audio_data, sample_rate)
        
        feature_vectors = np.array(feature_vectors)
        
        # Determine number of speakers
        if num_speakers is None:
            num_speakers = min(self._estimate_num_speakers(feature_vectors), 5)  # Max 5 speakers
        
        # Cluster features
        if len(feature_vectors) < num_speakers:
            num_speakers = len(feature_vectors)
        
        kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
        speaker_labels = kmeans.fit_predict(feature_vectors)
        
        # Create segments
        for (start_time, end_time), speaker_label in zip(time_windows, speaker_labels):
            segments.append(EnhancedSpeakerSegment(
                start_time=start_time,
                end_time=end_time,
                speaker_id=f"SPEAKER_{speaker_label + 1:02d}",
                confidence=0.7  # Medium confidence for acoustic method
            ))
        
        # Merge consecutive segments with same speaker
        segments = self._merge_consecutive_segments(segments)
        
        print(f"✅ 音響特徴量で {num_speakers} 話者を検出")
        return segments
    
    def _diarize_with_clustering(self, audio_data: np.ndarray, sample_rate: int,
                               num_speakers: Optional[int] = None) -> List[EnhancedSpeakerSegment]:
        """シンプルなクラスタリングによる話者識別."""
        print("🔬 クラスタリングによる話者識別を実行中...")
        
        # Simple energy-based segmentation
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        
        # Calculate energy and spectral centroid
        energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, hop_length=hop_length)[0]
        
        # Normalize features
        energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-8)
        spectral_centroid = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-8)
        
        # Combine features
        features = np.vstack([energy, spectral_centroid]).T
        
        # Determine number of speakers
        if num_speakers is None:
            num_speakers = 2  # Default to 2 speakers
        
        # Segment into 3-second windows
        window_frames = int(3.0 * sample_rate / hop_length)
        segments = []
        
        for i in range(0, len(features) - window_frames, window_frames // 2):
            window_features = features[i:i+window_frames]
            avg_features = np.mean(window_features, axis=0)
            
            # Simple speaker assignment based on spectral centroid
            if num_speakers == 2:
                speaker_id = "SPEAKER_01" if avg_features[1] < 0 else "SPEAKER_02"
            else:
                # Use k-means for more speakers
                if not hasattr(self, '_clustering_model'):
                    self._clustering_model = KMeans(n_clusters=num_speakers, random_state=42)
                    # Fit on all features
                    all_windows = []
                    for j in range(0, len(features) - window_frames, window_frames):
                        all_windows.append(np.mean(features[j:j+window_frames], axis=0))
                    self._clustering_model.fit(np.array(all_windows))
                
                speaker_idx = self._clustering_model.predict([avg_features])[0]
                speaker_id = f"SPEAKER_{speaker_idx + 1:02d}"
            
            start_time = i * hop_length / sample_rate
            end_time = (i + window_frames) * hop_length / sample_rate
            
            segments.append(EnhancedSpeakerSegment(
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_id,
                confidence=0.6  # Lower confidence for simple clustering
            ))
        
        # Merge consecutive segments
        segments = self._merge_consecutive_segments(segments)
        
        unique_speakers = len(set(seg.speaker_id for seg in segments))
        print(f"✅ クラスタリングで {unique_speakers} 話者を検出")
        return segments
    
    def _estimate_num_speakers(self, features: np.ndarray) -> int:
        """Estimate the number of speakers using elbow method."""
        if len(features) < 4:
            return 1
        
        max_k = min(5, len(features))
        inertias = []
        
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        if len(inertias) < 3:
            return 2
        
        # Calculate rate of change
        deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        if len(deltas) < 2:
            return 2
        
        # Find elbow (where improvement significantly decreases)
        ratios = [deltas[i] / deltas[i+1] if deltas[i+1] > 0 else 1 for i in range(len(deltas)-1)]
        
        # Return the k where improvement ratio is highest
        best_k = np.argmax(ratios) + 2  # +2 because we start from k=1 and take ratio
        return min(best_k, max_k)
    
    def _merge_consecutive_segments(self, segments: List[EnhancedSpeakerSegment]) -> List[EnhancedSpeakerSegment]:
        """Merge consecutive segments with the same speaker."""
        if not segments:
            return segments
        
        # Sort by start time
        segments.sort(key=lambda x: x.start_time)
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # If same speaker and close in time (within 0.5 seconds)
            if (current.speaker_id == last.speaker_id and 
                current.start_time - last.end_time <= 0.5):
                # Merge segments
                merged[-1] = EnhancedSpeakerSegment(
                    start_time=last.start_time,
                    end_time=current.end_time,
                    speaker_id=last.speaker_id,
                    confidence=(last.confidence + current.confidence) / 2
                )
            else:
                merged.append(current)
        
        return merged
    
    def _create_single_speaker_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[EnhancedSpeakerSegment]:
        """Create a single speaker segment for the entire audio."""
        duration = len(audio_data) / sample_rate
        return [EnhancedSpeakerSegment(
            start_time=0.0,
            end_time=duration,
            speaker_id="SPEAKER_01",
            confidence=0.5  # Low confidence for single speaker assumption
        )]
    
    def assign_speakers_to_transcription(self, 
                                       transcription_segments: List,
                                       speaker_segments: List[EnhancedSpeakerSegment],
                                       overlap_threshold: float = 0.5) -> List[Dict]:
        """
        Enhanced speaker assignment with overlap analysis.
        
        Args:
            transcription_segments: List of transcription segments
            speaker_segments: List of enhanced speaker segments
            overlap_threshold: Minimum overlap ratio to assign speaker
            
        Returns:
            List of transcription segments with enhanced speaker information
        """
        if not speaker_segments:
            print("⚠️  話者セグメントがありません。SPEAKER_UNKNOWN を割り当てます。")
            return [
                {
                    "start_time": getattr(seg, 'start_time', 0),
                    "end_time": getattr(seg, 'end_time', 0),
                    "text": getattr(seg, 'text', ''),
                    "confidence": getattr(seg, 'confidence', 0),
                    "speaker_id": "SPEAKER_UNKNOWN",
                    "speaker_confidence": 0.0
                }
                for seg in transcription_segments
            ]
        
        result = []
        for trans_seg in transcription_segments:
            # Get transcription segment times
            if hasattr(trans_seg, 'start_time'):
                trans_start = trans_seg.start_time
                trans_end = trans_seg.end_time
                trans_text = trans_seg.text
                trans_conf = trans_seg.confidence
            else:
                # Handle dict format
                trans_start = trans_seg.get('start_time', 0)
                trans_end = trans_seg.get('end_time', 0)
                trans_text = trans_seg.get('text', '')
                trans_conf = trans_seg.get('confidence', 0)
            
            # Find best matching speaker
            best_speaker = "SPEAKER_UNKNOWN"
            best_overlap = 0.0
            best_confidence = 0.0
            
            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(trans_start, speaker_seg.start_time)
                overlap_end = min(trans_end, speaker_seg.end_time)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    trans_duration = trans_end - trans_start
                    
                    if trans_duration > 0:
                        overlap_ratio = overlap_duration / trans_duration
                        
                        if overlap_ratio > best_overlap and overlap_ratio >= overlap_threshold:
                            best_overlap = overlap_ratio
                            best_speaker = speaker_seg.speaker_id
                            best_confidence = speaker_seg.confidence * overlap_ratio
            
            result.append({
                "start_time": trans_start,
                "end_time": trans_end,
                "text": trans_text,
                "confidence": trans_conf,
                "speaker_id": best_speaker,
                "speaker_confidence": best_confidence
            })
        
        # Log speaker assignment results
        unique_speakers = set(seg["speaker_id"] for seg in result)
        known_speakers = [s for s in unique_speakers if s != "SPEAKER_UNKNOWN"]
        
        if known_speakers:
            print(f"✅ 話者割り当て完了: {len(known_speakers)} 話者検出")
            for speaker in sorted(known_speakers):
                count = sum(1 for seg in result if seg["speaker_id"] == speaker)
                print(f"   - {speaker}: {count} セグメント")
        else:
            print("⚠️  話者識別ができませんでした。全セグメントが SPEAKER_UNKNOWN です。")
        
        return result

def get_speaker_statistics(segments: List[Dict]) -> Dict:
    """Get speaker statistics from segments."""
    speaker_stats = {}
    
    for seg in segments:
        speaker_id = seg.get("speaker_id", "SPEAKER_UNKNOWN")
        
        if speaker_id not in speaker_stats:
            speaker_stats[speaker_id] = {
                "segment_count": 0,
                "total_duration": 0.0,
                "total_words": 0,
                "avg_confidence": 0.0,
                "confidences": []
            }
        
        stats = speaker_stats[speaker_id]
        stats["segment_count"] += 1
        stats["total_duration"] += seg.get("end_time", 0) - seg.get("start_time", 0)
        stats["total_words"] += len(seg.get("text", "").split())
        
        conf = seg.get("confidence", 0)
        stats["confidences"].append(conf)
        stats["avg_confidence"] = sum(stats["confidences"]) / len(stats["confidences"])
    
    return speaker_stats