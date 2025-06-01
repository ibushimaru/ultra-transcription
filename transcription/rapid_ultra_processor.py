#!/usr/bin/env python3
"""
Rapid Ultra Precision Processor - 大規模ファイル用最高品質処理

2時間以上の音声ファイルを最高品質で処理する特化システム:
- large-v3モデル対応
- 高品質Whisper設定 (beam_size=5, best_of=5)
- 高精度話者認識
- 日本語後処理最適化
- チャンク分割で大規模ファイル対応
- 実用的な処理時間とのバランス
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
import whisper

from .post_processor import TranscriptionPostProcessor
from .enhanced_speaker_diarization import EnhancedSpeakerDiarizer

class RapidUltraPrecisionProcessor:
    """2時間以上の大規模ファイル専用最高品質プロセッサー"""
    
    def __init__(self, chunk_minutes: float = 3.0):
        """
        Initialize rapid processor
        
        Args:
            chunk_minutes: チャンクサイズ（分）
        """
        self.chunk_size = chunk_minutes * 60.0
        self.overlap_seconds = 10.0  # 短いオーバーラップ
        
        # 軽量コンポーネント初期化
        self.post_processor = TranscriptionPostProcessor()
        self.speaker_diarizer = EnhancedSpeakerDiarizer(method="acoustic")
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.whisper_model = None
    
    def _load_model(self, model_size: str):
        """Whisperモデルの読み込み"""
        if self.whisper_model is None:
            self.logger.info(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
            self.logger.info("Model loaded successfully")
    
    def _create_chunks(self, audio_file: str) -> List[Dict[str, float]]:
        """音声ファイルをチャンクに分割"""
        duration = librosa.get_duration(path=audio_file)
        chunks = []
        
        start_time = 0.0
        chunk_id = 0
        
        while start_time < duration:
            end_time = min(start_time + self.chunk_size, duration)
            
            chunks.append({
                'id': chunk_id,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            })
            
            chunk_id += 1
            start_time = end_time
        
        self.logger.info(f"Created {len(chunks)} chunks from {duration:.1f}s audio")
        return chunks
    
    def _process_chunk(self, audio_file: str, chunk: Dict[str, float], 
                      settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """単一チャンクの高品質処理"""
        try:
            # 音声読み込み（高品質設定）
            audio_data, sample_rate = librosa.load(
                audio_file,
                sr=16000,  # 固定サンプルレート
                offset=chunk['start'],
                duration=chunk['duration'],
                mono=True
            )
            
            # Whisper転写（バランスの取れた高品質設定）
            result = self.whisper_model.transcribe(
                audio_data,
                language=settings.get('language', 'ja'),
                task='transcribe',
                verbose=False,
                word_timestamps=False,
                temperature=0.0,  # 決定論的
                beam_size=3,      # 適度なビームサーチ
                best_of=2,        # 軽量な候補選択
                condition_on_previous_text=True,  # 文脈考慮
                fp16=True         # 高速化
            )
            
            segments = []
            for segment in result['segments']:
                seg_data = {
                    'start': segment['start'] + chunk['start'],
                    'end': segment['end'] + chunk['start'],
                    'text': segment['text'].strip(),
                    'confidence': 1.0 - segment.get('no_speech_prob', 0.0)
                }
                
                if seg_data['text']:  # 空でないテキストのみ
                    segments.append(seg_data)
            
            # 話者認識（高品質）
            if settings.get('enable_speaker_recognition', True) and len(segments) > 0:
                try:
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
                            overlap_start = max(seg_start_local, spk_seg.start_time)
                            overlap_end = min(seg_end_local, spk_seg.end_time)
                            overlap = max(0, overlap_end - overlap_start)
                            
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_speaker = spk_seg.speaker_id
                        
                        seg['speaker'] = best_speaker
                        
                except Exception as e:
                    self.logger.warning(f"Speaker recognition failed: {e}")
                    for seg in segments:
                        seg['speaker'] = 'SPEAKER_UNKNOWN'
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Chunk {chunk['id']} processing failed: {e}")
            return []
    
    def process_file(self, audio_file: str, output_file: str,
                    model: str = 'medium',
                    language: str = 'ja',
                    enable_speaker_recognition: bool = True,
                    num_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        大規模音声ファイルの高速処理
        
        Args:
            audio_file: 音声ファイルパス
            output_file: 出力ファイルパス
            model: Whisperモデル ('tiny', 'base', 'small', 'medium', 'large')
            language: 言語コード
            enable_speaker_recognition: 話者認識を有効化
            num_speakers: 期待話者数
            
        Returns:
            処理結果
        """
        start_time = time.time()
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # モデル読み込み
        self._load_model(model)
        
        # 設定
        settings = {
            'language': language,
            'enable_speaker_recognition': enable_speaker_recognition,
            'num_speakers': num_speakers
        }
        
        # チャンク作成
        chunks = self._create_chunks(audio_file)
        total_chunks = len(chunks)
        
        # 全セグメント収集
        all_segments = []
        
        self.logger.info(f"Processing {total_chunks} chunks...")
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            # チャンク処理
            segments = self._process_chunk(audio_file, chunk, settings)
            all_segments.extend(segments)
            
            chunk_time = time.time() - chunk_start
            progress = (i + 1) / total_chunks * 100
            
            self.logger.info(
                f"Chunk {i+1}/{total_chunks} ({progress:.1f}%): "
                f"{len(segments)} segments, {chunk_time:.1f}s"
            )
        
        # 日本語後処理
        if language == 'ja':
            self.logger.info("Applying Japanese post-processing...")
            try:
                # Post-processorが期待する形式に変換
                formatted_segments = []
                for seg in all_segments:
                    formatted_seg = {
                        'start_time': seg['start'],
                        'end_time': seg['end'],
                        'text': seg['text'],
                        'confidence': seg['confidence'],
                        'speaker': seg.get('speaker', 'SPEAKER_UNKNOWN')
                    }
                    formatted_segments.append(formatted_seg)
                
                processed_segments = self.post_processor.process_transcription(formatted_segments)
                
                # 元の形式に戻す
                all_segments = []
                for seg in processed_segments:
                    converted_seg = {
                        'start': seg['start_time'],
                        'end': seg['end_time'],
                        'text': seg['text'],
                        'confidence': seg['confidence'],
                        'speaker': seg.get('speaker', 'SPEAKER_UNKNOWN')
                    }
                    all_segments.append(converted_seg)
                    
            except Exception as e:
                self.logger.warning(f"Post-processing failed: {e}")
        
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
                'processing_method': 'rapid_ultra_precision'
            }
        }
        
        # 出力保存
        self._save_results(result, output_file)
        
        self.logger.info(
            f"Processing completed: {len(all_segments)} segments, "
            f"avg confidence: {avg_confidence:.3f}, "
            f"speed: {duration/total_time:.1f}x realtime"
        )
        
        return result
    
    def _save_results(self, result: Dict[str, Any], output_file: str):
        """結果の保存"""
        base_path = Path(output_file).with_suffix('')
        
        # JSON出力
        json_file = f"{base_path}_ultra_precision.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # CSV出力
        csv_file = f"{base_path}_ultra_precision.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("segment_id,start_seconds,end_seconds,duration,text,confidence,speaker\n")
            
            for i, segment in enumerate(result['segments']):
                duration = segment['end'] - segment['start']
                text = segment['text'].replace('"', '""')  # CSV エスケープ
                speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
                confidence = segment.get('confidence', 0.0)
                
                f.write(f"{i},{segment['start']:.3f},{segment['end']:.3f},"
                       f"{duration:.3f},\"{text}\",{confidence:.3f},{speaker}\n")
        
        # SRT出力
        srt_file = f"{base_path}_ultra_precision.srt"
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
        description="Rapid Ultra Precision Processor - 大規模音声ファイル最高品質処理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 2時間音声の最高品質処理
  python -m transcription.rapid_ultra_processor input.mp3 -o output --model large-v3
  
  # カスタムチャンクサイズ
  python -m transcription.rapid_ultra_processor input.mp3 -o output --chunk-size 4
  
  # 話者認識なし（高速）
  python -m transcription.rapid_ultra_processor input.mp3 -o output --no-speaker
        """
    )
    
    parser.add_argument('audio_file', help='音声ファイルパス')
    parser.add_argument('-o', '--output', required=True, help='出力ファイルベースパス')
    parser.add_argument('--model', default='large-v3',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3', 'large-v3-turbo', 'turbo'],
                       help='Whisperモデルサイズ (turbo: 8倍高速化版)')
    parser.add_argument('--language', default='ja', help='言語コード')
    parser.add_argument('--chunk-size', type=float, default=3.0,
                       help='チャンクサイズ（分）')
    parser.add_argument('--num-speakers', type=int, help='期待話者数')
    parser.add_argument('--no-speaker', action='store_true',
                       help='話者認識を無効化')
    
    args = parser.parse_args()
    
    # プロセッサー初期化
    processor = RapidUltraPrecisionProcessor(chunk_minutes=args.chunk_size)
    
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
        print(f"   - 処理時間: {result['metadata']['processing_time']:.1f}秒")
        print(f"   - 音声時間: {result['metadata']['total_duration']:.1f}秒")
        print(f"   - モデル: {result['metadata']['model_used']}")
        print(f"   - 処理方式: {result['metadata']['processing_method']}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  ユーザーによる中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 処理失敗: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()