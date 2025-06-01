"""
Real-time progress tracking for transcription tasks.
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class ProgressState:
    """Current progress state."""
    current_step: str
    step_progress: float
    overall_progress: float
    estimated_remaining: float
    elapsed_time: float


class ProgressTracker:
    """Track and display real-time progress for transcription."""
    
    def __init__(self, estimated_total_time: float):
        """
        Initialize progress tracker.
        
        Args:
            estimated_total_time: Estimated total processing time in seconds
        """
        self.estimated_total_time = estimated_total_time
        self.start_time = time.time()
        self.current_step = "初期化中"
        self.step_progress = 0.0
        self.overall_progress = 0.0
        self.is_running = False
        self.update_thread = None
        self.step_weights = {
            '音声読み込み': 0.05,
            '前処理': 0.15,
            '文字起こし': 0.60,
            '後処理': 0.10,
            'スピーカー識別': 0.08,
            '出力保存': 0.02
        }
        self.completed_steps = set()
    
    def start(self):
        """Start progress tracking."""
        self.is_running = True
        self.start_time = time.time()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def stop(self):
        """Stop progress tracking."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
    
    def update_step(self, step_name: str, progress: float = 0.0):
        """
        Update current step and progress.
        
        Args:
            step_name: Name of current processing step
            progress: Progress within current step (0.0-1.0)
        """
        self.current_step = step_name
        self.step_progress = max(0.0, min(1.0, progress))
        
        # Calculate overall progress
        completed_weight = sum(
            self.step_weights.get(step, 0.05) 
            for step in self.completed_steps
        )
        
        current_weight = self.step_weights.get(step_name, 0.05)
        current_contribution = current_weight * self.step_progress
        
        self.overall_progress = completed_weight + current_contribution
        self.overall_progress = max(0.0, min(1.0, self.overall_progress))
    
    def complete_step(self, step_name: str):
        """Mark a step as completed."""
        self.completed_steps.add(step_name)
        self.update_step(step_name, 1.0)
    
    def get_current_state(self) -> ProgressState:
        """Get current progress state."""
        elapsed = time.time() - self.start_time
        
        if self.overall_progress > 0:
            estimated_total = elapsed / self.overall_progress
            estimated_remaining = max(0, estimated_total - elapsed)
        else:
            estimated_remaining = self.estimated_total_time
        
        return ProgressState(
            current_step=self.current_step,
            step_progress=self.step_progress,
            overall_progress=self.overall_progress,
            estimated_remaining=estimated_remaining,
            elapsed_time=elapsed
        )
    
    def _update_loop(self):
        """Background thread for progress updates."""
        last_update = 0
        
        while self.is_running:
            current_time = time.time()
            
            # Update every 2 seconds
            if current_time - last_update >= 2.0:
                state = self.get_current_state()
                self._display_progress(state)
                last_update = current_time
            
            time.sleep(0.5)
    
    def _display_progress(self, state: ProgressState):
        """Display current progress."""
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * state.overall_progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format times
        elapsed_str = self._format_time(state.elapsed_time)
        remaining_str = self._format_time(state.estimated_remaining)
        
        # Display progress line
        print(f"\r🔄 {state.current_step} [{bar}] {state.overall_progress:.1%} | "
              f"経過: {elapsed_str} | 残り: {remaining_str}", end='', flush=True)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def display_final_summary(self):
        """Display final processing summary."""
        final_state = self.get_current_state()
        print(f"\n✅ 処理完了! 総時間: {self._format_time(final_state.elapsed_time)}")
        
        # Compare with estimate
        accuracy = abs(final_state.elapsed_time - self.estimated_total_time) / self.estimated_total_time
        if accuracy < 0.2:
            accuracy_msg = "🎯 予想通り"
        elif accuracy < 0.5:
            accuracy_msg = "📊 まずまず"
        else:
            accuracy_msg = "📈 予想外"
        
        print(f"予想精度: {accuracy_msg} (予想: {self._format_time(self.estimated_total_time)})")


class EnhancedProgressCallback:
    """Enhanced progress callback for transcription engines."""
    
    def __init__(self, tracker: ProgressTracker):
        """Initialize with progress tracker."""
        self.tracker = tracker
        self.current_step = ""
    
    def audio_loading(self, progress: float = 0.0):
        """Audio loading progress."""
        self.tracker.update_step("音声読み込み", progress)
    
    def preprocessing(self, substep: str = "", progress: float = 0.0):
        """Preprocessing progress."""
        step_name = f"前処理{f' ({substep})' if substep else ''}"
        self.tracker.update_step(step_name, progress)
    
    def transcription(self, progress: float = 0.0, segments_done: int = 0, total_segments: int = 0):
        """Transcription progress."""
        if total_segments > 0:
            detail = f" ({segments_done}/{total_segments})"
        else:
            detail = ""
        
        step_name = f"文字起こし{detail}"
        self.tracker.update_step(step_name, progress)
    
    def post_processing(self, substep: str = "", progress: float = 0.0):
        """Post-processing progress."""
        step_name = f"後処理{f' ({substep})' if substep else ''}"
        self.tracker.update_step(step_name, progress)
    
    def speaker_diarization(self, progress: float = 0.0):
        """Speaker diarization progress."""
        self.tracker.update_step("スピーカー識別", progress)
    
    def output_saving(self, format_name: str = "", progress: float = 0.0):
        """Output saving progress."""
        step_name = f"出力保存{f' ({format_name})' if format_name else ''}"
        self.tracker.update_step(step_name, progress)