"""
Advanced precision enhancement system with multiple accuracy improvement strategies.
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from .faster_transcriber import FasterTranscriber
from .enhanced_audio_processor import EnhancedAudioProcessor


@dataclass
class ModelPerformance:
    """Performance metrics for a specific model."""
    model_name: str
    confidence: float
    segments: List[Dict]
    processing_time: float
    memory_usage: float


class EnsembleTranscriber:
    """Ensemble transcription using multiple models for higher accuracy."""
    
    def __init__(self, models: List[str] = None, language: str = 'ja', device: str = 'cpu'):
        """
        Initialize ensemble transcriber.
        
        Args:
            models: List of model sizes to use ['medium', 'large', 'large-v3-turbo']
            language: Target language
            device: Processing device
        """
        self.models = models or ['base', 'medium']  # Start conservative
        self.language = language
        self.device = device
        self.transcribers = {}
        
        # Initialize transcribers for each model
        for model in self.models:
            try:
                print(f"🔧 Loading {model} model...")
                self.transcribers[model] = FasterTranscriber(
                    model_size=model,
                    language=language,
                    device=device
                )
                print(f"✅ {model} model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load {model} model: {e}")
                # Remove failed model from list
                if model in self.models:
                    self.models.remove(model)
    
    def transcribe_with_ensemble(self, 
                               audio_data: np.ndarray, 
                               sample_rate: int,
                               min_confidence: float = 0.3,
                               voting_method: str = 'confidence_weighted') -> Tuple[List[Dict], Dict]:
        """
        Transcribe audio using ensemble of models.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            min_confidence: Minimum confidence threshold
            voting_method: Method for combining results ('confidence_weighted', 'majority')
            
        Returns:
            Tuple of (final_segments, ensemble_metadata)
        """
        print(f"🎭 Ensemble transcription using {len(self.models)} models...")
        
        model_results = {}
        performance_metrics = {}
        
        # Transcribe with each model
        for model_name in self.models:
            if model_name not in self.transcribers:
                continue
                
            start_time = time.time()
            print(f"📝 Transcribing with {model_name}...")
            
            try:
                transcriber = self.transcribers[model_name]
                segments = transcriber.process_transcription(
                    audio_data,
                    sample_rate,
                    filter_confidence=True,
                    filter_fillers=True,
                    min_confidence=min_confidence
                )
                
                processing_time = time.time() - start_time
                
                # Convert to dict format
                segment_dicts = []
                confidences = []
                for seg in segments:
                    seg_dict = {
                        'start_time': seg.start_time,
                        'end_time': seg.end_time,
                        'text': seg.text,
                        'confidence': seg.confidence
                    }
                    segment_dicts.append(seg_dict)
                    confidences.append(seg.confidence)
                
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                model_results[model_name] = segment_dicts
                performance_metrics[model_name] = ModelPerformance(
                    model_name=model_name,
                    confidence=avg_confidence,
                    segments=segment_dicts,
                    processing_time=processing_time,
                    memory_usage=0.0  # TODO: Implement memory tracking
                )
                
                print(f"✅ {model_name}: {len(segments)} segments, {avg_confidence:.3f} avg confidence")
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                continue
        
        if not model_results:
            raise RuntimeError("All models failed to transcribe")
        
        # Combine results using ensemble method
        if voting_method == 'confidence_weighted':
            final_segments = self._confidence_weighted_ensemble(model_results, performance_metrics)
        elif voting_method == 'majority':
            final_segments = self._majority_vote_ensemble(model_results, performance_metrics)
        else:
            # Fallback: use best performing model
            best_model = max(performance_metrics.keys(), 
                           key=lambda k: performance_metrics[k].confidence)
            final_segments = model_results[best_model]
        
        # Quality check: if ensemble result is significantly worse than best individual, use best individual
        best_individual_confidence = max(perf.confidence for perf in performance_metrics.values())
        ensemble_confidence = np.mean([seg['confidence'] for seg in final_segments]) if final_segments else 0
        
        if ensemble_confidence < best_individual_confidence * 0.85:  # 15% degradation threshold
            print(f"⚠️  Ensemble quality degradation detected, using best individual model")
            best_model = max(performance_metrics.keys(), 
                           key=lambda k: performance_metrics[k].confidence)
            final_segments = model_results[best_model]
        
        # Create ensemble metadata
        ensemble_metadata = {
            'models_used': list(model_results.keys()),
            'voting_method': voting_method,
            'model_performances': {
                name: {
                    'confidence': perf.confidence,
                    'processing_time': perf.processing_time,
                    'segment_count': len(perf.segments)
                }
                for name, perf in performance_metrics.items()
            },
            'ensemble_improvement': self._calculate_ensemble_improvement(
                performance_metrics, final_segments
            )
        }
        
        return final_segments, ensemble_metadata
    
    def _confidence_weighted_ensemble(self, 
                                    model_results: Dict[str, List[Dict]], 
                                    performance_metrics: Dict[str, ModelPerformance]) -> List[Dict]:
        """Combine results using confidence-weighted voting."""
        print("🗳️  Applying confidence-weighted ensemble voting...")
        
        # Get model weights based on overall performance
        weights = {}
        total_weight = 0
        for model_name, perf in performance_metrics.items():
            weight = perf.confidence
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_weight if total_weight > 0 else 1
        
        # For simplicity, use the highest-weighted model's segmentation as base
        best_model = max(weights.keys(), key=lambda k: weights[k])
        base_segments = model_results[best_model]
        
        # Enhance with confidence-weighted text selection
        enhanced_segments = []
        for base_seg in base_segments:
            best_text = base_seg['text']
            best_confidence = base_seg['confidence']  # Keep original confidence
            best_model_for_seg = best_model
            
            # Check if other models have overlapping segments
            for model_name, segments in model_results.items():
                if model_name == best_model:
                    continue
                    
                # Find best overlapping segment
                best_overlap = 0
                best_seg = None
                for seg in segments:
                    overlap = self._calculate_time_overlap(base_seg, seg)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_seg = seg
                
                # If good overlap and higher confidence, use this segment
                if best_seg and best_overlap > 0.3 and best_seg['confidence'] > best_confidence:
                    best_text = best_seg['text']
                    best_confidence = best_seg['confidence']
                    best_model_for_seg = model_name
            
            # Apply ensemble boost for segments agreed upon by multiple models
            ensemble_boost = 1.0
            if best_model_for_seg != best_model:
                ensemble_boost = 1.05  # Small boost for cross-model agreement
            
            enhanced_segments.append({
                'start_time': base_seg['start_time'],
                'end_time': base_seg['end_time'],
                'text': best_text,
                'confidence': min(best_confidence * ensemble_boost, 1.0)
            })
        
        print(f"✅ Ensemble voting completed: {len(enhanced_segments)} segments")
        return enhanced_segments
    
    def _majority_vote_ensemble(self, 
                              model_results: Dict[str, List[Dict]], 
                              performance_metrics: Dict[str, ModelPerformance]) -> List[Dict]:
        """Combine results using majority voting."""
        print("🗳️  Applying majority vote ensemble...")
        
        # Use best performing model as base structure
        best_model = max(performance_metrics.keys(), 
                        key=lambda k: performance_metrics[k].confidence)
        base_segments = model_results[best_model]
        
        # For each segment, collect votes from other models
        enhanced_segments = []
        for base_seg in base_segments:
            text_votes = [base_seg['text']]
            confidence_votes = [base_seg['confidence']]
            
            # Collect votes from other models
            for model_name, segments in model_results.items():
                if model_name == best_model:
                    continue
                    
                # Find best overlapping segment
                best_overlap = 0
                best_seg = None
                for seg in segments:
                    overlap = self._calculate_time_overlap(base_seg, seg)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_seg = seg
                
                if best_seg and best_overlap > 0.3:  # 30% overlap threshold
                    text_votes.append(best_seg['text'])
                    confidence_votes.append(best_seg['confidence'])
            
            # Select majority text or highest confidence if no majority
            if len(text_votes) > 1:
                text_counter = Counter(text_votes)
                most_common = text_counter.most_common(1)[0]
                if most_common[1] > 1:  # Has majority
                    final_text = most_common[0]
                else:
                    # No majority, use highest confidence
                    max_idx = np.argmax(confidence_votes)
                    final_text = text_votes[max_idx]
            else:
                final_text = base_seg['text']
            
            # Average confidence
            final_confidence = np.mean(confidence_votes)
            
            enhanced_segments.append({
                'start_time': base_seg['start_time'],
                'end_time': base_seg['end_time'],
                'text': final_text,
                'confidence': final_confidence
            })
        
        print(f"✅ Majority voting completed: {len(enhanced_segments)} segments")
        return enhanced_segments
    
    def _calculate_time_overlap(self, seg1: Dict, seg2: Dict) -> float:
        """Calculate time overlap ratio between two segments."""
        start1, end1 = seg1['start_time'], seg1['end_time']
        start2, end2 = seg2['start_time'], seg2['end_time']
        
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        seg1_duration = end1 - start1
        
        return overlap_duration / seg1_duration if seg1_duration > 0 else 0.0
    
    def _calculate_ensemble_improvement(self, 
                                      performance_metrics: Dict[str, ModelPerformance],
                                      final_segments: List[Dict]) -> Dict:
        """Calculate improvement metrics from ensemble."""
        if not final_segments:
            return {}
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean([seg['confidence'] for seg in final_segments])
        
        # Best individual model confidence
        best_individual = max(perf.confidence for perf in performance_metrics.values())
        
        improvement = {
            'ensemble_confidence': ensemble_confidence,
            'best_individual_confidence': best_individual,
            'absolute_improvement': ensemble_confidence - best_individual,
            'relative_improvement': ((ensemble_confidence - best_individual) / best_individual * 100) if best_individual > 0 else 0
        }
        
        return improvement


class AdvancedVAD:
    """Advanced Voice Activity Detection for better segmentation."""
    
    def __init__(self, threshold: float = 0.5, min_speech_duration: int = 200):
        """
        Initialize advanced VAD.
        
        Args:
            threshold: Voice activity threshold
            min_speech_duration: Minimum speech duration in ms
        """
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
    
    def apply_advanced_vad(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply advanced VAD to improve chunk boundaries."""
        try:
            import torch
            import torchaudio
            
            # Use torchaudio VAD if available
            if hasattr(torchaudio.transforms, 'Vad'):
                vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
                audio_tensor = torch.from_numpy(audio_data).float()
                vad_audio = vad(audio_tensor)
                return vad_audio.numpy()
            else:
                # Fallback to simple energy-based VAD
                return self._energy_based_vad(audio_data, sample_rate)
                
        except ImportError:
            # Fallback to simple energy-based VAD
            return self._energy_based_vad(audio_data, sample_rate)
    
    def _energy_based_vad(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Simple energy-based VAD as fallback."""
        # Calculate frame energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)
        
        # Normalize energy
        energy = np.array(energy)
        if len(energy) > 0:
            energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        
        # Apply threshold
        speech_frames = energy > self.threshold
        
        # Convert back to audio samples
        speech_audio = np.zeros_like(audio_data)
        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                start_idx = i * hop_length
                end_idx = min(start_idx + frame_length, len(audio_data))
                speech_audio[start_idx:end_idx] = audio_data[start_idx:end_idx]
        
        return speech_audio


class PrecisionEnhancer:
    """Main class for applying all precision enhancement strategies."""
    
    def __init__(self, use_ensemble: bool = True, 
                 ensemble_models: List[str] = None,
                 use_advanced_vad: bool = True):
        """
        Initialize precision enhancer.
        
        Args:
            use_ensemble: Whether to use ensemble transcription
            ensemble_models: List of models for ensemble
            use_advanced_vad: Whether to use advanced VAD
        """
        self.use_ensemble = use_ensemble
        self.ensemble_models = ensemble_models or ['base', 'medium']
        self.use_advanced_vad = use_advanced_vad
        
        # Initialize components
        if self.use_ensemble:
            self.ensemble_transcriber = None  # Initialize on demand
        
        if self.use_advanced_vad:
            self.vad = AdvancedVAD()
    
    def enhance_transcription(self, 
                            audio_data: np.ndarray, 
                            sample_rate: int,
                            language: str = 'ja',
                            device: str = 'cpu',
                            min_confidence: float = 0.3) -> Tuple[List[Dict], Dict]:
        """
        Apply all precision enhancement strategies.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            language: Target language
            device: Processing device
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (enhanced_segments, enhancement_metadata)
        """
        enhancement_metadata = {
            'techniques_applied': [],
            'performance_improvements': {}
        }
        
        # Apply advanced VAD if enabled
        if self.use_advanced_vad:
            print("🎙️  Applying advanced VAD...")
            audio_data = self.vad.apply_advanced_vad(audio_data, sample_rate)
            enhancement_metadata['techniques_applied'].append('advanced_vad')
        
        # Apply ensemble transcription if enabled
        if self.use_ensemble:
            print("🎭 Applying ensemble transcription...")
            
            # Initialize ensemble transcriber on demand
            if self.ensemble_transcriber is None:
                self.ensemble_transcriber = EnsembleTranscriber(
                    models=self.ensemble_models,
                    language=language,
                    device=device
                )
            
            enhanced_segments, ensemble_metadata = self.ensemble_transcriber.transcribe_with_ensemble(
                audio_data, sample_rate, min_confidence
            )
            
            enhancement_metadata['techniques_applied'].append('ensemble_transcription')
            enhancement_metadata['ensemble_details'] = ensemble_metadata
            
        else:
            # Fallback to single model transcription
            print("📝 Using single model transcription...")
            transcriber = FasterTranscriber(
                model_size=self.ensemble_models[0] if self.ensemble_models else 'base',
                language=language,
                device=device
            )
            
            segments = transcriber.process_transcription(
                audio_data, sample_rate, 
                filter_confidence=True,
                filter_fillers=True,
                min_confidence=min_confidence
            )
            
            # Convert to dict format
            enhanced_segments = []
            for seg in segments:
                enhanced_segments.append({
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'text': seg.text,
                    'confidence': seg.confidence
                })
        
        return enhanced_segments, enhancement_metadata