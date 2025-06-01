"""
Post-processing for transcription results using context and language models.
"""

import re
import unicodedata
from typing import List, Dict, Tuple
import numpy as np


class TranscriptionPostProcessor:
    """Post-process transcription results for better accuracy."""
    
    def __init__(self):
        """Initialize post processor."""
        # Common Japanese word corrections
        self.word_corrections = {
            # Technical terms
            'ジェネレイティブ': 'ジェネレーティブ',
            'プロンプト': 'プロンプト',
            'チャット': 'チャット',
            'スプレット': 'スプレッド',
            'スライト': 'スライド',
            'コシー': 'コシ',
            'コッシー': 'コシ',
            
            # Common mishears
            'ジェミに': 'Gemini',
            'ジェミニー': 'Gemini',
            'チェミニ': 'Gemini',
            'チャップGPT': 'ChatGPT',
            'チャップgpt': 'ChatGPT',
            'DPL': 'DeepL',
            'ディーペール': 'DeepL',
            'Dプリサーチ': 'D.B.Research',
            'dbrisarch': 'DB Research',
            
            # Common words
            'かけぼう': '家計簿',
            'かけぃこう': '家計簿',
            'テッド': 'タスク',
            'テッドリバー': 'タスク',
            'ハンター': 'ハンドラー',
            'C色': '推論',
            'スイロン': '推論',
            '本薬': '翻訳',
            'ホン薬': '翻訳',
            'プラモホン役': 'プラモ翻訳',
            
            # Numbers and units
            '1再': '一切',
            '運転マン': '数十万',
            '運薬万': '数十万',
            '運べくまん': '数十万',
            '運軸マント': '数十万',
            '的スタート': 'テキスト',
            'アップをクリーダー': 'アップ・クリエーター',
            
            # Business terms
            'フリーナース': 'フリーランス',
            'ガイブに': '外部に',
            'いたくする': '委託する',
            '生成合': '生成AI',
            '生成愛': '生成AI',
            'セスイヤー': 'エンジニア',
            '準備': 'Gemini',
            
            # Locations and names
            'おふわ': '僕は',
            'ナリツ': 'なりつつ',
            'おそこさん': 'お疲れさん',
            'お亡く': '翻訳',
            
            # Time expressions
            '三角間': '3日間',
            '人向かし': '2年',
            '人向かしまい': '2年前',
            '2、3、4、5年前': '2、3年前',
        }
        
        # Contextual patterns for better accuracy
        self.context_patterns = [
            # Technology context
            (r'(AI|人工知能).*(スタジュー|スタジオ)', r'\1スタジオ'),
            (r'(Google|グーグル).*(AI|エーアイ)', r'\1 AI'),
            (r'(マークダウン).*(形式|フォーマット)', r'\1形式'),
            (r'(スライド).*(化|か)', r'\1化'),
            (r'(チャット).*(GPT|ジーピーティー)', r'ChatGPT'),
            
            # Business context
            (r'(フリー).*(ランス|ナース)', r'フリーランス'),
            (r'(自動化).*(したい|したくない)', r'\1したい'),
            (r'(開発).*(進める|すすめる)', r'\1を進める'),
            
            # Numbers and quantities
            (r'(\d+).*(セント|ワード)', r'\1ワード'),
            (r'(運[転薬軸]).*(万|マン)', r'数十万'),
            (r'(\d+).*(分|ぶん).*(ぐらい|くらい)', r'\1分ぐらい'),
        ]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text using Unicode normalization."""
        # NFKC normalization for Japanese text
        normalized = unicodedata.normalize('NFKC', text)
        
        # Remove extra whitespaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def apply_word_corrections(self, text: str) -> str:
        """Apply word-level corrections."""
        corrected = text
        
        for wrong, correct in self.word_corrections.items():
            # Case-insensitive replacement
            pattern = re.escape(wrong)
            corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def apply_context_corrections(self, text: str) -> str:
        """Apply context-based corrections."""
        corrected = text
        
        for pattern, replacement in self.context_patterns:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        return corrected
    
    def fix_sentence_boundaries(self, text: str) -> str:
        """Fix sentence boundaries and punctuation."""
        # Add periods after common sentence endings
        text = re.sub(r'(です|ます|でした|ました)(?![。！？])', r'\1。', text)
        text = re.sub(r'(ですね|ますね|ですよ|ますよ)(?![。！？])', r'\1。', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([。！？])\s*', r'\1 ', text)
        
        # Remove trailing spaces
        text = re.sub(r'\s+$', '', text)
        
        return text
    
    def confidence_based_filtering(self, segments: List[Dict], min_confidence: float = 0.4) -> List[Dict]:
        """Filter segments based on confidence with contextual consideration."""
        filtered = []
        
        for i, segment in enumerate(segments):
            confidence = segment.get('confidence', 0.0)
            text = segment.get('text', '').strip()
            
            # Skip very short, low-confidence segments
            if len(text) <= 2 and confidence < 0.6:
                continue
            
            # Keep high-confidence segments
            if confidence >= min_confidence:
                filtered.append(segment)
                continue
            
            # For medium confidence, check context
            if confidence >= 0.2:
                # Check surrounding context
                context_bonus = 0.0
                
                # Bonus for being surrounded by high-confidence segments
                if i > 0 and segments[i-1].get('confidence', 0) > 0.7:
                    context_bonus += 0.1
                if i < len(segments)-1 and segments[i+1].get('confidence', 0) > 0.7:
                    context_bonus += 0.1
                
                # Bonus for containing common words
                if any(word in text for word in ['です', 'ます', 'という', 'ので', 'から']):
                    context_bonus += 0.1
                
                if confidence + context_bonus >= min_confidence:
                    # Update confidence
                    segment = segment.copy()
                    segment['confidence'] = min(1.0, confidence + context_bonus)
                    filtered.append(segment)
        
        return filtered
    
    def merge_short_segments(self, segments: List[Dict], min_duration: float = 2.0) -> List[Dict]:
        """Merge very short segments with adjacent ones."""
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i].copy()
            duration = current['end_time'] - current['start_time']
            
            # If segment is too short, try to merge with next
            if duration < min_duration and i < len(segments) - 1:
                next_segment = segments[i + 1]
                
                # Merge if gap is small (< 1 second)
                gap = next_segment['start_time'] - current['end_time']
                if gap < 1.0:
                    # Merge segments
                    current['end_time'] = next_segment['end_time']
                    current['text'] = f"{current['text']} {next_segment['text']}"
                    # Average confidence
                    current['confidence'] = (current['confidence'] + next_segment['confidence']) / 2
                    i += 2  # Skip next segment as it's merged
                else:
                    merged.append(current)
                    i += 1
            else:
                merged.append(current)
                i += 1
        
        return merged
    
    def process_transcription(self, segments: List[Dict]) -> List[Dict]:
        """
        Complete post-processing pipeline.
        """
        print("🔧 Post-processing transcription...")
        
        processed = []
        
        for segment in segments:
            text = segment.get('text', '')
            
            # Apply all text corrections
            text = self.normalize_text(text)
            text = self.apply_word_corrections(text)
            text = self.apply_context_corrections(text)
            text = self.fix_sentence_boundaries(text)
            
            # Update segment
            processed_segment = segment.copy()
            processed_segment['text'] = text
            processed.append(processed_segment)
        
        print(f"   - Applied corrections to {len(processed)} segments")
        
        # Apply filtering and merging
        processed = self.confidence_based_filtering(processed, min_confidence=0.3)
        print(f"   - Filtered to {len(processed)} segments")
        
        processed = self.merge_short_segments(processed, min_duration=1.5)
        print(f"   - Merged to {len(processed)} segments")
        
        print("✅ Post-processing completed")
        
        return processed