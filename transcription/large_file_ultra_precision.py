#!/usr/bin/env python3
"""
Large File Ultra Precision Processor

大規模ファイル用の最高品質処理システム:
- large-v3モデル使用
- 話者認識統合
- アンサンブル処理（複数モデル）なし（処理時間短縮）
- 高品質でも実用的な処理時間
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
import numpy as np

from .transcriber import Transcriber
from .post_processor import TranscriptionPostProcessor
from .enhanced_speaker_diarization import EnhancedSpeakerDiarizer

class LargeFileUltraPrecision:
    """大規模ファイル用最高品質プロセッサー（実用的処理時間）"""
    
    def __init__(self, chunk_minutes: float = 8.0):
        """
        Initialize large file ultra precision processor
        
        Args:
            chunk_minutes: チャンクサイズ（分）
        """
        self.chunk_size = chunk_minutes * 60.0
        self.overlap_seconds = 15.0
        
        # コンポーネント初期化
        self.post_processor = TranscriptionPostProcessor()
        self.speaker_diarizer = EnhancedSpeakerDiarizer(method="acoustic")
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.transcriber = None
    
    def _load_transcriber(self, model_size: str, language: str):
        """高品質Transcriber読み込み"""
        if self.transcriber is None:
            self.logger.info(f"Loading high-quality transcriber: {model_size}")
            self.transcriber = Transcriber(model_size=model_size, language=language)
            self.logger.info("High-quality transcriber loaded")
    
    def _create_chunks(self, audio_file: str) -> List[Dict[str, float]]:
        """音声ファイルをチャンクに分割"""
        duration = librosa.get_duration(path=audio_file)
        chunks = []
        
        start_time = 0.0
        chunk_id = 0
        
        while start_time < duration:
            end_time = min(start_time + self.chunk_size, duration)
            # オーバーラップ追加
            if end_time < duration:
                actual_end_time = min(end_time + self.overlap_seconds, duration)
            else:
                actual_end_time = end_time
            
            chunks.append({
                'id': chunk_id,
                'start': start_time,
                'end': actual_end_time,
                'content_end': end_time,  # 実際のコンテンツ終了時間
                'duration': actual_end_time - start_time
            })
            
            chunk_id += 1
            start_time = end_time
        
        self.logger.info(f"Created {len(chunks)} chunks from {duration:.1f}s audio")
        return chunks
    
    def _process_chunk(self, audio_file: str, chunk: Dict[str, float], 
                      settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """単一チャンクの最高品質処理"""
        try:
            self.logger.info(f"Processing chunk {chunk['id']} ({chunk['start']/60:.1f}-{chunk['end']/60:.1f}min)")
            
            # 音声読み込み（高品質設定）
            audio_data, sample_rate = librosa.load(
                audio_file,
                sr=16000,
                offset=chunk['start'],
                duration=chunk['duration'],
                mono=True
            )
            
            # 高品質転写
            segment_results = self.transcriber.transcribe_audio(audio_data, sample_rate)
            
            # 形式変換
            segments = []
            for seg in segment_results:
                seg_data = {
                    'start': seg.start_time + chunk['start'],
                    'end': seg.end_time + chunk['start'],
                    'text': seg.text.strip(),
                    'confidence': seg.confidence
                }
                
                # オーバーラップ部分をフィルタ（コンテンツ範囲内のみ）
                if seg_data['start'] < chunk['content_end'] and seg_data['text']:
                    # 終了時間もコンテンツ範囲内に調整
                    if seg_data['end'] > chunk['content_end']:
                        seg_data['end'] = chunk['content_end']
                    segments.append(seg_data)
            
            # 話者認識（高精度）
            if settings.get('enable_speaker_recognition', True) and len(segments) > 0:
                try:
                    self.logger.info(f"Applying speaker recognition to chunk {chunk['id']}")
                    speaker_segments = self.speaker_diarizer.diarize_audio(
                        audio_data, sample_rate,
                        num_speakers=settings.get('num_speakers')
                    )
                    
                    # 話者情報マージ
                    for seg in segments:
                        seg_start_local = seg['start'] - chunk['start']
                        seg_end_local = seg['end'] - chunk['start']
                        
                        best_speaker = 'SPEAKER_UNKNOWN'
                        max_overlap = 0.0
                        
                        for spk_seg in speaker_segments:
                            overlap_start = max(seg_start_local, spk_seg['start'])
                            overlap_end = min(seg_end_local, spk_seg['end'])
                            overlap = max(0, overlap_end - overlap_start)
                            
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_speaker = spk_seg.get('speaker', 'SPEAKER_UNKNOWN')
                        
                        seg['speaker'] = best_speaker
                        
                except Exception as e:
                    self.logger.warning(f"Speaker recognition failed for chunk {chunk['id']}: {e}")
                    for seg in segments:
                        seg['speaker'] = 'SPEAKER_UNKNOWN'
            
            self.logger.info(f"Chunk {chunk['id']} completed: {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Chunk {chunk['id']} processing failed: {e}")
            return []
    
    def _merge_chunks(self, all_chunks_results: List[List[Dict]], chunks: List[Dict]) -> List[Dict]:
        """チャンク結果をマージ（オーバーラップ処理）"""
        if not all_chunks_results:
            return []
        
        merged_segments = []
        
        for i, chunk_segments in enumerate(all_chunks_results):
            if i == 0:
                # 最初のチャンクはそのまま追加
                merged_segments.extend(chunk_segments)
            else:
                # 後続チャンクはオーバーラップを除去
                chunk_start = chunks[i]['start']
                prev_chunk_content_end = chunks[i-1]['content_end']
                
                for segment in chunk_segments:
                    # オーバーラップ範囲外のセグメントのみ追加
                    if segment['start'] >= prev_chunk_content_end:
                        merged_segments.append(segment)
        
        return merged_segments
    
    def process_file(self, audio_file: str, output_file: str,
                    model: str = 'large-v3',
                    language: str = 'ja',
                    enable_speaker_recognition: bool = True,
                    num_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        大規模音声ファイルの最高品質処理
        
        Args:
            audio_file: 音声ファイルパス
            output_file: 出力ファイルパス
            model: Whisperモデル (推奨: 'large-v3')
            language: 言語コード
            enable_speaker_recognition: 話者認識を有効化
            num_speakers: 期待話者数
            
        Returns:
            処理結果
        """
        start_time = time.time()
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Transcriber読み込み
        self._load_transcriber(model, language)
        
        # 設定
        settings = {
            'language': language,
            'enable_speaker_recognition': enable_speaker_recognition,
            'num_speakers': num_speakers
        }
        
        # チャンク作成
        chunks = self._create_chunks(audio_file)
        total_chunks = len(chunks)
        
        # 全チャンク処理
        all_chunks_results = []
        
        self.logger.info(f"Processing {total_chunks} chunks with ultra precision...")
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            # チャンク処理
            segments = self._process_chunk(audio_file, chunk, settings)
            all_chunks_results.append(segments)
            
            chunk_time = time.time() - chunk_start
            progress = (i + 1) / total_chunks * 100
            
            self.logger.info(
                f"Chunk {i+1}/{total_chunks} ({progress:.1f}%): "
                f"{len(segments)} segments, {chunk_time:.1f}s"
            )
        
        # チャンクマージ
        self.logger.info("Merging chunks...")
        all_segments = self._merge_chunks(all_chunks_results, chunks)
        
        # 日本語後処理
        if language == 'ja':
            self.logger.info("Applying Japanese post-processing...")
            try:
                all_segments = self.post_processor.process_transcription(all_segments)
            except Exception as e:
                self.logger.warning(f"Post-processing failed: {e}")
        
        # 話者一貫性処理
        if enable_speaker_recognition:
            self.logger.info("Applying speaker consistency...")
            all_segments = self._apply_speaker_consistency(all_segments)
        
        # 統計計算
        total_time = time.time() - start_time
        duration = librosa.get_duration(path=audio_file)
        avg_confidence = np.mean([seg.get('confidence', 0.0) for seg in all_segments]) if all_segments else 0.0
        
        # 結果作成
        result = {
            'format_version': '2.0',
            'segments': all_segments,
            'metadata': {
                'original_file': audio_file,
                'total_duration': duration,
                'total_segments': len(all_segments),
                'total_chunks': total_chunks,
                'processing_time': total_time,
                'real_time_factor': duration / total_time if total_time > 0 else 0,
                'average_confidence': avg_confidence,
                'model_used': model,
                'language': language,
                'processing_method': 'large_file_ultra_precision',
                'speaker_recognition_enabled': enable_speaker_recognition
            }
        }
        
        # 出力保存
        self._save_results(result, output_file)
        
        self.logger.info(
            f"Ultra precision processing completed: {len(all_segments)} segments, "
            f"avg confidence: {avg_confidence:.3f}, "
            f"speed: {duration/total_time:.1f}x realtime, "
            f"total time: {total_time/60:.1f}min"
        )
        
        return result
    
    def _apply_speaker_consistency(self, segments: List[Dict]) -> List[Dict]:
        """話者一貫性アルゴリズム適用"""
        if len(segments) < 2:
            return segments
        
        # 短いセグメントのマージ
        i = 1
        while i < len(segments):
            current = segments[i]
            previous = segments[i-1]
            
            # 非常に短いセグメント（2秒未満）で同じ話者の場合、マージ
            if (current['end'] - current['start'] < 2.0 and 
                current.get('speaker') == previous.get('speaker') and
                current['start'] - previous['end'] < 1.0):
                
                # マージ
                previous['end'] = current['end']
                previous['text'] += ' ' + current['text']
                previous['confidence'] = (previous['confidence'] + current['confidence']) / 2
                
                segments.pop(i)
            else:
                i += 1
        
        return segments
    
    def _save_results(self, result: Dict[str, Any], output_file: str):
        """結果の保存"""
        base_path = Path(output_file).with_suffix('')
        
        # JSON出力
        json_file = f"{base_path}_ultra.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # CSV出力
        csv_file = f"{base_path}_ultra.csv"
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
        srt_file = f"{base_path}_ultra.srt"
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
        description="Large File Ultra Precision Processor - 大規模音声ファイル最高品質処理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 2時間音声の最高品質処理
  python -m transcription.large_file_ultra_precision input.mp3 -o output
  
  # カスタム設定
  python -m transcription.large_file_ultra_precision input.mp3 -o output --model large-v3 --chunk-size 10
        """
    )
    
    parser.add_argument('audio_file', help='音声ファイルパス')
    parser.add_argument('-o', '--output', required=True, help='出力ファイルベースパス')
    parser.add_argument('--model', default='large-v3',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3'],
                       help='Whisperモデルサイズ')
    parser.add_argument('--language', default='ja', help='言語コード')
    parser.add_argument('--chunk-size', type=float, default=8.0,
                       help='チャンクサイズ（分）')
    parser.add_argument('--num-speakers', type=int, help='期待話者数')
    parser.add_argument('--no-speaker', action='store_true',
                       help='話者認識を無効化')
    
    args = parser.parse_args()
    
    # プロセッサー初期化
    processor = LargeFileUltraPrecision(chunk_minutes=args.chunk_size)
    
    try:
        # ファイル処理
        result = processor.process_file(
            audio_file=args.audio_file,
            output_file=args.output,
            model=args.model,
            language=args.language,
            enable_speaker_recognition=not args.no_speaker,
            num_speakers=args.num_speakers
        )
        
        print(f"\n✅ 最高品質処理完了!")
        print(f"📊 結果:")
        print(f"   - 総セグメント数: {result['metadata']['total_segments']}")
        print(f"   - 平均信頼度: {result['metadata']['average_confidence']:.1%}")
        print(f"   - 処理速度: {result['metadata']['real_time_factor']:.1f}x リアルタイム")
        print(f"   - 処理時間: {result['metadata']['processing_time']/60:.1f}分")
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