# 開発者ガイド - 音声文字起こしシステム

## 目次
1. [開発環境セットアップ](#開発環境セットアップ)
2. [コードベース構造](#コードベース構造)
3. [核心技術の詳解](#核心技術の詳解)
4. [API仕様](#api仕様)
5. [新機能の追加方法](#新機能の追加方法)
6. [テスト方法](#テスト方法)
7. [パフォーマンス最適化](#パフォーマンス最適化)
8. [デプロイメント](#デプロイメント)

## 開発環境セットアップ

### 1. 開発依存関係
```bash
# 開発用追加パッケージ
pip install pytest pytest-cov black flake8 mypy
pip install jupyter notebook  # 分析用
```

### 2. 環境変数設定
```bash
# .env ファイル作成
echo "HF_TOKEN=your_huggingface_token" > .env
echo "CUDA_VISIBLE_DEVICES=0" >> .env  # GPU使用時
```

### 3. プロジェクト構造
```
transcription/
├── transcription/           # メインパッケージ
│   ├── __init__.py
│   ├── main.py             # 基本エントリーポイント
│   ├── turbo_enhanced_main.py  # Turbo最適化版
│   ├── maximum_precision_main.py  # 最高精度版
│   └── [各種コンポーネント]
├── docs/                   # ドキュメント
├── benchmarks/             # パフォーマンステスト
├── configs/                # 設定ファイル
├── test_outputs/          # テスト結果アーカイブ
└── testdata/              # テスト用音声ファイル
```

## コードベース構造

### 核心コンポーネント

#### 1. 音声処理レイヤー
```python
# enhanced_audio_processor.py
class EnhancedAudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def advanced_preprocess_audio(self, 
                                 file_path: str,
                                 enable_noise_reduction: bool = True,
                                 enable_speech_enhancement: bool = True,
                                 memory_efficient: bool = False) -> Tuple[np.ndarray, int]:
        \"\"\"
        高度音声前処理
        
        Args:
            file_path: 音声ファイルパス
            enable_noise_reduction: ノイズ除去有効化
            enable_speech_enhancement: 音声強化有効化
            memory_efficient: メモリ効率モード
            
        Returns:
            (audio_data, sample_rate): 処理済み音声データ
        \"\"\"
```

#### 2. 転写エンジンレイヤー
```python
# faster_transcriber.py
class FasterTranscriber:
    def __init__(self, model_size: str = \"medium\", device: str = \"cpu\"):
        self.model = WhisperModel(model_size, device=device)
    
    def process_transcription(self, 
                            audio_data: np.ndarray,
                            sample_rate: int,
                            language: str = \"ja\") -> List[TranscriptionSegment]:
        \"\"\"
        音声転写処理
        
        Returns:
            List[TranscriptionSegment]: 転写結果セグメント
        \"\"\"
```

#### 3. アンサンブル処理レイヤー
```python
# precision_enhancer.py
class EnsembleTranscriber:
    def ensemble_transcribe(self,
                          audio_data: np.ndarray,
                          sample_rate: int,
                          models: List[str],
                          voting_method: str = \"confidence_weighted\") -> List[dict]:
        \"\"\"
        複数モデルアンサンブル転写
        
        Args:
            models: 使用モデルリスト [\"base\", \"medium\"]
            voting_method: 投票方式
            
        Returns:
            最適化された転写結果
        \"\"\"
```

### データ構造

#### TranscriptionSegment
```python
@dataclass
class TranscriptionSegment:
    start_time: float
    end_time: float
    text: str
    confidence: float
    speaker_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            \"start_time\": self.format_timestamp(self.start_time),
            \"end_time\": self.format_timestamp(self.end_time),
            \"text\": self.text,
            \"confidence\": round(self.confidence, 3),
            \"speaker_id\": self.speaker_id or \"SPEAKER_UNKNOWN\"
        }
```

#### ProcessingMetadata
```python
@dataclass
class ProcessingMetadata:
    input_file: str
    model_size: str
    engine: str
    processing_time_seconds: float
    average_confidence: float
    total_segments: int
    techniques_applied: Dict[str, bool]
```

## 核心技術の詳解

### 1. Turbo最適化アルゴリズム

#### チャンクサイズ最適化
```python
def optimize_chunk_size(audio_duration: float, turbo_mode: bool) -> int:
    \"\"\"
    Whisper Large-v3 Turbo向け最適チャンクサイズ計算
    
    研究結果に基づく最適化:
    - 10-15秒: Turbo処理に最適
    - VADとの組み合わせで精度維持
    \"\"\"
    if turbo_mode:
        if audio_duration < 300:  # 5分未満
            return 10
        else:
            return 15
    return 30  # 標準モード
```

#### メモリ効率処理
```python
def memory_efficient_processing(audio_data: np.ndarray, 
                              chunk_size: int,
                              overlap: float = 2.0) -> Iterator[np.ndarray]:
    \"\"\"
    メモリ使用量を抑制したチャンク処理
    
    - ジェネレータパターンで逐次処理
    - オーバーラップ処理で境界精度維持
    - ガベージコレクション最適化
    \"\"\"
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size+overlap]
        yield chunk
        del chunk  # 明示的メモリ解放
```

### 2. アンサンブル投票アルゴリズム

#### 信頼度重み付け投票
```python
def confidence_weighted_voting(segments_list: List[List[dict]]) -> List[dict]:
    \"\"\"
    複数モデル結果の信頼度重み付け統合
    
    アルゴリズム:
    1. 時間軸アラインメント
    2. 信頼度による重み計算
    3. テキスト類似度評価
    4. 最適解選択
    \"\"\"
    aligned_segments = align_temporal_segments(segments_list)
    final_segments = []
    
    for segment_group in aligned_segments:
        weights = [seg['confidence'] for seg in segment_group]
        text_candidates = [seg['text'] for seg in segment_group]
        
        # 信頼度とテキスト類似度の組み合わせ評価
        best_segment = select_best_candidate(segment_group, weights, text_candidates)
        final_segments.append(best_segment)
    
    return final_segments
```

### 3. 日本語特化後処理

#### フィラーワード除去
```python
JAPANESE_FILLERS = [
    r\"\\b(えー+|あー+|うー+ん?)\\b\",
    r\"\\b(そのー+|なんか|まあ)\\b\",
    r\"\\b(っていうか|みたいな)\\b\"
]

def remove_japanese_fillers(text: str) -> str:
    \"\"\"
    日本語フィラーワード除去
    
    - 正規表現による高精度除去
    - 文脈保持アルゴリズム
    - 自然な文章流れ維持
    \"\"\"
    for pattern in JAPANESE_FILLERS:
        text = re.sub(pattern, \"\", text, flags=re.IGNORECASE)
    return clean_whitespace(text)
```

## API仕様

### 1. コマンドラインAPI

#### Turbo Enhanced
```bash
python -m transcription.turbo_enhanced_main [AUDIO_FILE] [OPTIONS]

OPTIONS:
  --model TEXT              Whisperモデル [tiny|base|small|medium|large-v3]
  --turbo-mode             Turbo最適化有効化
  --realtime-mode          リアルタイム処理モード
  --chunk-size INTEGER     チャンクサイズ（秒）
  --language TEXT          言語コード
  --min-confidence FLOAT   最小信頼度閾値
  --device TEXT            デバイス [cpu|cuda]
  --hf-token TEXT          HuggingFaceトークン
  --format TEXT            出力形式 [all|json|csv|txt|srt]
  --output TEXT            出力ファイルベース名
  --auto-confirm           自動確認（バッチ処理用）
```

#### Maximum Precision
```bash
python -m transcription.maximum_precision_main [AUDIO_FILE] [OPTIONS]

OPTIONS:
  --use-ensemble           アンサンブル処理有効化
  --ensemble-models TEXT   アンサンブルモデルリスト
  --voting-method TEXT     投票方式 [confidence_weighted|majority]
  --use-advanced-vad       高度VAD使用
```

### 2. Python API

#### 基本使用例
```python
from transcription.turbo_enhanced_main import turbo_enhanced_transcribe
from transcription.enhanced_audio_processor import EnhancedAudioProcessor

# 音声処理
processor = EnhancedAudioProcessor()
audio_data, sample_rate = processor.advanced_preprocess_audio(
    \"audio.mp3\",
    enable_noise_reduction=True,
    memory_efficient=True
)

# 転写実行
segments = turbo_enhanced_transcribe(
    audio_file=\"audio.mp3\",
    turbo_mode=True,
    model=\"base\",
    language=\"ja\"
)
```

#### カスタム処理パイプライン
```python
from transcription.faster_transcriber import FasterTranscriber
from transcription.post_processor import TranscriptionPostProcessor

# カスタムパイプライン構築
transcriber = FasterTranscriber(model_size=\"medium\", device=\"cpu\")
post_processor = TranscriptionPostProcessor()

# 処理実行
segments = transcriber.process_transcription(audio_data, sample_rate)
processed_segments = post_processor.process_transcription(segments)
```

## 新機能の追加方法

### 1. 新しい音声前処理の追加

```python
# enhanced_audio_processor.py に追加
def custom_audio_enhancement(self, audio_data: np.ndarray) -> np.ndarray:
    \"\"\"
    カスタム音声強化処理
    \"\"\"
    # 新しいアルゴリズム実装
    enhanced_audio = your_algorithm(audio_data)
    return enhanced_audio

# advanced_preprocess_audio メソッドに統合
if enable_custom_enhancement:
    audio_data = self.custom_audio_enhancement(audio_data)
```

### 2. 新しい出力形式の追加

```python
# output_formatter.py に追加
def save_as_xml(self, segments: List[dict], output_path: str) -> str:
    \"\"\"
    XML形式での保存
    \"\"\"
    xml_content = self.generate_xml(segments)
    with open(f\"{output_path}.xml\", \"w\", encoding=\"utf-8\") as f:
        f.write(xml_content)
    return f\"{output_path}.xml\"

# save_all_formats メソッドに追加
saved_files[\"xml\"] = self.save_as_xml(segments, output_path)
```

### 3. 新しいモデルの統合

```python
# faster_transcriber.py を参考に新クラス作成
class CustomTranscriber:
    def __init__(self, model_path: str):
        self.model = load_custom_model(model_path)
    
    def transcribe(self, audio_data: np.ndarray) -> List[TranscriptionSegment]:
        # カスタムモデル処理
        results = self.model.process(audio_data)
        return self.format_results(results)
```

## テスト方法

### 1. 単体テスト

```python
# tests/test_audio_processor.py
import pytest
from transcription.enhanced_audio_processor import EnhancedAudioProcessor

def test_audio_loading():
    processor = EnhancedAudioProcessor()
    audio_data, sr = processor.load_audio(\"test_audio.wav\")
    assert sr == 16000
    assert len(audio_data) > 0

def test_noise_reduction():
    processor = EnhancedAudioProcessor()
    noisy_audio = generate_noisy_audio()
    clean_audio = processor.advanced_noise_reduction(noisy_audio, 16000)
    assert calculate_snr(clean_audio) > calculate_snr(noisy_audio)
```

### 2. 統合テスト

```python
# tests/test_integration.py
def test_full_pipeline():
    \"\"\"完全パイプラインのエンドツーエンドテスト\"\"\"
    result = turbo_enhanced_transcribe(
        audio_file=\"tests/data/test_audio.wav\",
        turbo_mode=True,
        auto_confirm=True
    )
    assert result[\"average_confidence\"] > 0.7
    assert len(result[\"segments\"]) > 0
```

### 3. パフォーマンステスト

```python
# tests/test_performance.py
import time

def test_turbo_speed():
    \"\"\"Turboモードの速度検証\"\"\"
    start_time = time.time()
    result = turbo_enhanced_transcribe(
        \"tests/data/10min_audio.wav\",
        turbo_mode=True
    )
    processing_time = time.time() - start_time
    
    # 10分音声を2分以内で処理
    assert processing_time < 120
```

### 4. テスト実行

```bash
# 全テスト実行
pytest tests/

# カバレッジ付き
pytest --cov=transcription tests/

# 特定テスト
pytest tests/test_audio_processor.py::test_noise_reduction -v
```

## パフォーマンス最適化

### 1. メモリ最適化

```python
# メモリ使用量監視
import psutil
import gc

def monitor_memory_usage():
    \"\"\"メモリ使用量監視\"\"\"
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f\"Memory usage: {memory_mb:.1f} MB\")

# 明示的ガベージコレクション
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 2. 処理速度最適化

```python
# NumPy最適化
import numpy as np
from numba import jit

@jit(nopython=True)
def fast_audio_processing(audio_data: np.ndarray) -> np.ndarray:
    \"\"\"JITコンパイル済み高速音声処理\"\"\"
    # NumPy最適化アルゴリズム
    return processed_audio

# 並列処理
from concurrent.futures import ThreadPoolExecutor

def parallel_chunk_processing(chunks: List[np.ndarray]) -> List[dict]:
    \"\"\"並列チャンク処理\"\"\"
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_chunk, chunks))
    return results
```

### 3. GPU最適化

```python
# CUDA利用最適化
def optimize_gpu_usage():
    \"\"\"GPU使用最適化\"\"\"
    if torch.cuda.is_available():
        # GPU warmup
        dummy_tensor = torch.randn(1000, 1000).cuda()
        _ = torch.mm(dummy_tensor, dummy_tensor)
        del dummy_tensor
        torch.cuda.empty_cache()
```

## デプロイメント

### 1. Docker化

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY transcription/ ./transcription/
COPY configs/ ./configs/

EXPOSE 8000
CMD [\"python\", \"-m\", \"transcription.api_server\"]
```

### 2. API サーバー

```python
# api_server.py
from fastapi import FastAPI, UploadFile, File
from transcription.turbo_enhanced_main import turbo_enhanced_transcribe

app = FastAPI()

@app.post(\"/transcribe/\")
async def transcribe_audio(file: UploadFile = File(...)):
    \"\"\"音声ファイル転写API\"\"\"
    # ファイル保存
    temp_path = save_uploaded_file(file)
    
    # 転写実行
    result = turbo_enhanced_transcribe(
        audio_file=temp_path,
        turbo_mode=True,
        auto_confirm=True
    )
    
    # 一時ファイル削除
    os.remove(temp_path)
    
    return result
```

### 3. バッチ処理スクリプト

```python
# scripts/batch_process.py
import os
import glob
from transcription.turbo_enhanced_main import turbo_enhanced_transcribe

def batch_transcribe(input_dir: str, output_dir: str):
    \"\"\"ディレクトリ内全音声ファイルのバッチ処理\"\"\"
    audio_files = glob.glob(f\"{input_dir}/*.wav\") + glob.glob(f\"{input_dir}/*.mp3\")
    
    for audio_file in audio_files:
        output_name = os.path.join(output_dir, os.path.basename(audio_file))
        
        try:
            turbo_enhanced_transcribe(
                audio_file=audio_file,
                output=output_name,
                turbo_mode=True,
                auto_confirm=True
            )
            print(f\"✅ Completed: {audio_file}\")
        except Exception as e:
            print(f\"❌ Failed: {audio_file} - {e}\")

if __name__ == \"__main__\":
    batch_transcribe(\"input/\", \"output/\")
```

## 貢献ガイドライン

### 1. コードスタイル
- Black フォーマッター使用
- Type hints 必須
- Docstring は Google スタイル

### 2. プルリクエスト
1. フィーチャーブランチ作成
2. テスト追加
3. ドキュメント更新
4. レビュー依頼

### 3. 課題報告
- 再現手順明記
- 環境情報添付
- ログファイル添付

---

このガイドに従って開発を進めることで、高品質で保守性の高いコードベースを維持できます。