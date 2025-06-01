#!/usr/bin/env python3
"""
Segmented Large File Processor - 大容量音声ファイル分割処理システム

長時間音声ファイルを分割して個別処理し、結果をマージ:
- 10分セグメントに自動分割
- 各セグメントを独立処理
- 結果の自動マージ
- タイムアウト問題解決
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile

import librosa
import soundfile as sf
import numpy as np

from .large_file_ultra_precision import LargeFileUltraPrecision

class SegmentedProcessor:
    """大容量ファイル用分割処理システム"""
    
    def __init__(self, segment_minutes: float = 10.0, overlap_seconds: float = 5.0):
        """
        Initialize segmented processor
        
        Args:
            segment_minutes: セグメント長（分）
            overlap_seconds: セグメント間オーバーラップ（秒）
        """
        self.segment_length = segment_minutes * 60.0
        self.overlap_length = overlap_seconds
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.processor = LargeFileUltraPrecision(chunk_minutes=2.0)  # 小さなチャンク
    
    def _create_segments(self, audio_file: str, temp_dir: str) -> List[Dict[str, Any]]:
        """音声ファイルをセグメントに分割"""
        duration = librosa.get_duration(path=audio_file)
        audio_data, sample_rate = librosa.load(audio_file, sr=None)
        
        segments = []
        start_time = 0.0
        segment_id = 0
        
        while start_time < duration:
            # セグメント範囲計算
            end_time = min(start_time + self.segment_length, duration)
            
            # オーバーラップ追加（最後以外）
            if end_time < duration:
                actual_end_time = min(end_time + self.overlap_length, duration)
            else:
                actual_end_time = end_time
            
            # セグメント音声抽出
            start_sample = int(start_time * sample_rate)
            end_sample = int(actual_end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # セグメントファイル保存
            segment_file = os.path.join(temp_dir, f"segment_{segment_id:03d}.wav")
            sf.write(segment_file, segment_audio, sample_rate)
            
            segments.append({
                'id': segment_id,
                'file': segment_file,
                'start_time': start_time,
                'end_time': actual_end_time,
                'content_end_time': end_time,  # 実際のコンテンツ終了時間
                'duration': actual_end_time - start_time
            })
            
            self.logger.info(f"Segment {segment_id}: {start_time/60:.1f}-{actual_end_time/60:.1f}min ({segment_file})")
            
            segment_id += 1
            start_time = end_time
        
        self.logger.info(f"Created {len(segments)} segments from {duration/60:.1f}min audio")
        return segments
    
    def _process_segment(self, segment: Dict[str, Any], output_dir: str,
                        model: str, language: str, enable_speaker: bool) -> Optional[Dict[str, Any]]:
        """単一セグメントの処理"""
        try:
            segment_output = os.path.join(output_dir, f"segment_{segment['id']:03d}")
            
            self.logger.info(f"Processing segment {segment['id']} ({segment['duration']/60:.1f}min)...")
            
            # セグメント処理
            result = self.processor.process_file(
                audio_file=segment['file'],
                output_file=segment_output,
                model=model,
                language=language,
                enable_speaker_recognition=enable_speaker,
                num_speakers=None
            )
            
            # セグメント情報を結果に追加
            result['segment_info'] = segment
            
            return result
            
        except Exception as e:
            self.logger.error(f"Segment {segment['id']} processing failed: {e}")
            return None
    
    def _merge_segments(self, segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """セグメント結果をマージ"""
        if not segment_results:
            return {'segments': [], 'metadata': {}}
        
        merged_segments = []
        total_duration = 0.0
        total_processing_time = 0.0
        
        for i, result in enumerate(segment_results):
            if not result or 'segments' not in result:
                continue
            
            segment_info = result['segment_info']
            offset = segment_info['start_time']
            content_end = segment_info['content_end_time']
            
            # セグメント内の転写結果を調整
            for seg in result['segments']:
                # タイムスタンプ調整
                adjusted_start = seg['start'] + offset
                adjusted_end = seg['end'] + offset
                
                # オーバーラップ範囲外をフィルタ
                if i > 0:  # 最初のセグメント以外
                    prev_content_end = segment_results[i-1]['segment_info']['content_end_time']
                    if adjusted_start < prev_content_end:
                        continue  # オーバーラップ部分をスキップ
                
                # コンテンツ範囲内のみ
                if adjusted_start < content_end:
                    # 終了時間もコンテンツ範囲内に調整
                    if adjusted_end > content_end:
                        adjusted_end = content_end
                    
                    merged_seg = {
                        'start': adjusted_start,
                        'end': adjusted_end,
                        'text': seg['text'],
                        'confidence': seg['confidence'],
                        'speaker': seg.get('speaker', 'SPEAKER_UNKNOWN')
                    }
                    merged_segments.append(merged_seg)
            
            # メタデータ累積
            if 'metadata' in result:
                total_duration = max(total_duration, segment_info['content_end_time'])
                total_processing_time += result['metadata'].get('processing_time', 0)
        
        # 統計計算
        avg_confidence = np.mean([seg['confidence'] for seg in merged_segments]) if merged_segments else 0.0
        
        return {
            'format_version': '2.0',
            'segments': merged_segments,
            'metadata': {
                'total_duration': total_duration,
                'total_segments': len(merged_segments),
                'total_processing_time': total_processing_time,
                'real_time_factor': total_duration / total_processing_time if total_processing_time > 0 else 0,
                'average_confidence': avg_confidence,
                'processing_method': 'segmented_ultra_precision',
                'segment_count': len(segment_results)
            }
        }
    
    def process_file(self, audio_file: str, output_file: str,
                    model: str = 'large-v3',
                    language: str = 'ja',
                    enable_speaker_recognition: bool = False) -> Dict[str, Any]:
        """
        大容量音声ファイルの分割処理
        
        Args:
            audio_file: 音声ファイルパス
            output_file: 出力ファイルパス
            model: Whisperモデル
            language: 言語コード
            enable_speaker_recognition: 話者認識を有効化
            
        Returns:
            処理結果
        """
        start_time = time.time()
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # 一時ディレクトリ作成
        with tempfile.TemporaryDirectory() as temp_dir:
            # セグメント分割
            self.logger.info("Splitting audio into segments...")
            segments = self._create_segments(audio_file, temp_dir)
            
            # 各セグメント処理
            segment_results = []
            output_dir = Path(output_file).parent
            
            for i, segment in enumerate(segments):
                self.logger.info(f"Processing segment {i+1}/{len(segments)}...")
                
                result = self._process_segment(
                    segment, str(output_dir), model, language, enable_speaker_recognition
                )
                
                if result:
                    segment_results.append(result)
                    self.logger.info(f"Segment {i+1} completed: {len(result['segments'])} segments")
                else:
                    self.logger.warning(f"Segment {i+1} failed")
        
        # 結果マージ
        self.logger.info("Merging segment results...")
        final_result = self._merge_segments(segment_results)
        
        # 追加メタデータ
        total_time = time.time() - start_time
        final_result['metadata'].update({
            'original_file': audio_file,
            'model_used': model,
            'language': language,
            'total_processing_time': total_time,
            'segments_processed': len(segments),
            'segments_successful': len(segment_results)
        })
        
        # 結果保存
        self._save_results(final_result, output_file)
        
        self.logger.info(
            f"Segmented processing completed: {len(final_result['segments'])} segments, "
            f"avg confidence: {final_result['metadata']['average_confidence']:.3f}, "
            f"total time: {total_time/60:.1f}min"
        )
        
        return final_result
    
    def _save_results(self, result: Dict[str, Any], output_file: str):
        """結果の保存"""
        base_path = Path(output_file).with_suffix('')
        
        # JSON出力
        json_file = f"{base_path}_segmented.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # CSV出力
        csv_file = f"{base_path}_segmented.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("segment_id,start_seconds,end_seconds,duration,text,confidence,speaker\n")
            
            for i, segment in enumerate(result['segments']):
                duration = segment['end'] - segment['start']
                text = segment['text'].replace('"', '""')
                speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
                confidence = segment.get('confidence', 0.0)
                
                f.write(f"{i},{segment['start']:.3f},{segment['end']:.3f},"
                       f"{duration:.3f},\"{text}\",{confidence:.3f},{speaker}\n")
        
        # SRT出力
        srt_file = f"{base_path}_segmented.srt"
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments']):
                start_time = self._seconds_to_srt_time(segment['start'])
                end_time = self._seconds_to_srt_time(segment['end'])
                
                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
        
        self.logger.info(f"Results saved: {json_file}, {csv_file}, {srt_file}")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """秒をSRT時間形式に変換"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def main():
    """コマンドラインインターフェース"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Segmented Processor - 大容量音声ファイル分割処理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 45分音声の分割処理
  python -m transcription.segmented_processor input.mp3 -o output --model large-v3
  
  # カスタムセグメントサイズ
  python -m transcription.segmented_processor input.mp3 -o output --segment-size 15
        """
    )
    
    parser.add_argument('audio_file', help='音声ファイルパス')
    parser.add_argument('-o', '--output', required=True, help='出力ファイルベースパス')
    parser.add_argument('--model', default='large-v3-turbo',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo'],
                       help='Whisperモデルサイズ (turbo: 8倍高速化版)')
    parser.add_argument('--language', default='ja', help='言語コード')
    parser.add_argument('--segment-size', type=float, default=10.0,
                       help='セグメントサイズ（分）')
    parser.add_argument('--overlap', type=float, default=5.0,
                       help='オーバーラップ（秒）')
    parser.add_argument('--enable-speaker', action='store_true',
                       help='話者認識を有効化')
    
    args = parser.parse_args()
    
    # プロセッサー初期化
    processor = SegmentedProcessor(
        segment_minutes=args.segment_size,
        overlap_seconds=args.overlap
    )
    
    try:
        # ファイル処理
        result = processor.process_file(
            audio_file=args.audio_file,
            output_file=args.output,
            model=args.model,
            language=args.language,
            enable_speaker_recognition=args.enable_speaker
        )
        
        print(f"\n✅ 分割処理完了!")
        print(f"📊 結果:")
        print(f"   - 総セグメント数: {result['metadata']['total_segments']}")
        print(f"   - 平均信頼度: {result['metadata']['average_confidence']:.1%}")
        print(f"   - 分割数: {result['metadata']['segment_count']}")
        print(f"   - 処理時間: {result['metadata']['total_processing_time']/60:.1f}分")
        print(f"   - 音声時間: {result['metadata']['total_duration']/60:.1f}分")
        print(f"   - モデル: {result['metadata']['model_used']}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  ユーザーによる中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 処理失敗: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()