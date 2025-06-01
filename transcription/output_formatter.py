"""
Output formatting for transcription results.
"""

import json
import csv
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path


class OutputFormatter:
    """Handle formatting and saving transcription results."""
    
    def __init__(self):
        """Initialize output formatter."""
        pass
    
    def format_timestamp(self, seconds: float) -> str:
        """
        Format seconds to HH:MM:SS.mmm format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def prepare_output_data(self, transcription_segments: List[Dict]) -> List[Dict]:
        """
        Prepare transcription data for output.
        
        Args:
            transcription_segments: List of transcription segments with speaker info
            
        Returns:
            Formatted data for output
        """
        output_data = []
        
        for i, segment in enumerate(transcription_segments):
            output_data.append({
                "segment_id": i + 1,
                "start_time": self.format_timestamp(segment["start_time"]),
                "end_time": self.format_timestamp(segment["end_time"]),
                "start_seconds": round(segment["start_time"], 3),
                "end_seconds": round(segment["end_time"], 3),
                "duration": round(segment["end_time"] - segment["start_time"], 3),
                "speaker_id": segment["speaker_id"],
                "text": segment["text"],
                "confidence": round(segment["confidence"], 3)
            })
        
        return output_data
    
    def save_as_json(self, data: List[Dict], output_path: str, metadata: Dict = None):
        """
        Save transcription as JSON file.
        
        Args:
            data: Transcription data
            output_path: Output file path
            metadata: Additional metadata to include
        """
        output = {
            "metadata": metadata or {},
            "segments": data,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    def save_as_csv(self, data: List[Dict], output_path: str):
        """
        Save transcription as CSV file.
        
        Args:
            data: Transcription data
            output_path: Output file path
        """
        if not data:
            return
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def save_as_txt(self, data: List[Dict], output_path: str):
        """
        Save transcription as readable text file.
        
        Args:
            data: Transcription data
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("音声文字起こし結果\\n")
            f.write("=" * 50 + "\\n\\n")
            
            current_speaker = None
            
            for segment in data:
                # Add speaker header if speaker changes
                if segment["speaker_id"] != current_speaker:
                    current_speaker = segment["speaker_id"]
                    f.write(f"\\n[{current_speaker}]\\n")
                
                # Write segment with timestamp
                f.write(f"[{segment['start_time']} - {segment['end_time']}] ")
                f.write(f"({segment['confidence']:.2f}) ")
                f.write(f"{segment['text']}\\n")
    
    def save_as_srt(self, data: List[Dict], output_path: str):
        """
        Save transcription as SRT subtitle file.
        
        Args:
            data: Transcription data
            output_path: Output file path
        """
        def seconds_to_srt_time(seconds: float) -> str:
            """Convert seconds to SRT time format."""
            td = timedelta(seconds=seconds)
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            milliseconds = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(data, 1):
                start_time = seconds_to_srt_time(segment["start_seconds"])
                end_time = seconds_to_srt_time(segment["end_seconds"])
                
                f.write(f"{i}\\n")
                f.write(f"{start_time} --> {end_time}\\n")
                f.write(f"[{segment['speaker_id']}] {segment['text']}\\n")
                f.write("\\n")
    
    def save_all_formats(self, transcription_segments: List[Dict], 
                        base_output_path: str, metadata: Dict = None):
        """
        Save transcription in all supported formats.
        
        Args:
            transcription_segments: List of transcription segments with speaker info
            base_output_path: Base path for output files (without extension)
            metadata: Additional metadata to include
        """
        # Prepare data
        output_data = self.prepare_output_data(transcription_segments)
        
        # Get base path
        base_path = Path(base_output_path)
        
        # Save in different formats
        self.save_as_json(output_data, f"{base_path}.json", metadata)
        self.save_as_csv(output_data, f"{base_path}.csv")
        self.save_as_txt(output_data, f"{base_path}.txt")
        self.save_as_srt(output_data, f"{base_path}.srt")
        
        return {
            "json": f"{base_path}.json",
            "csv": f"{base_path}.csv", 
            "txt": f"{base_path}.txt",
            "srt": f"{base_path}.srt"
        }
    
    def print_summary(self, data: List[Dict], audio_duration: float = None):
        """
        Print transcription summary to console.
        
        Args:
            data: Transcription data
            audio_duration: Total audio duration in seconds
        """
        if not data:
            print("No transcription data to display.")
            return
        
        print("\\n" + "=" * 60)
        print("文字起こし結果サマリー")
        print("=" * 60)
        
        # Basic statistics
        total_segments = len(data)
        total_text_length = sum(len(segment["text"]) for segment in data)
        avg_confidence = sum(segment["confidence"] for segment in data) / total_segments
        
        speakers = set(segment["speaker_id"] for segment in data)
        
        print(f"総セグメント数: {total_segments}")
        print(f"総文字数: {total_text_length}")
        print(f"平均信頼度: {avg_confidence:.2f}")
        print(f"検出スピーカー数: {len(speakers)}")
        print(f"スピーカー: {', '.join(sorted(speakers))}")
        
        if audio_duration:
            print(f"音声長: {self.format_timestamp(audio_duration)}")
        
        print("\\n" + "-" * 60)
        print("セグメント詳細 (最初の5つ):")
        print("-" * 60)
        
        for i, segment in enumerate(data[:5]):
            print(f"{i+1}. [{segment['start_time']}-{segment['end_time']}] "
                  f"[{segment['speaker_id']}] "
                  f"({segment['confidence']:.2f}) "
                  f"{segment['text']}")
        
        if total_segments > 5:
            print(f"... ({total_segments - 5} more segments)")
        
        print("=" * 60)