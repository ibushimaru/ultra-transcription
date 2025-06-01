# API リファレンス - 音声文字起こしシステム

## 目次
1. [コマンドライン API](#コマンドライン-api)
2. [Python API](#python-api)
3. [データ形式](#データ形式)
4. [エラーハンドリング](#エラーハンドリング)
5. [設定オプション](#設定オプション)

## コマンドライン API

### 1. Turbo Enhanced モード

#### 基本構文
```bash
python -m transcription.turbo_enhanced_main [AUDIO_FILE] [OPTIONS]
```

#### 引数

| 引数 | 型 | 必須 | 説明 |
|------|-----|------|------|
| `AUDIO_FILE` | str | ✅ | 音声ファイルパス（MP3, WAV, M4A等） |

#### オプション

| オプション | 型 | デフォルト | 説明 |
|------------|-----|------------|------|
| `--output, -o` | str | 自動生成 | 出力ファイルのベース名 |
| `--model, -m` | str | `large-v3` | Whisperモデル（tiny/base/small/medium/large-v3） |
| `--turbo-mode` | flag | `True` | Turbo最適化モード有効化 |
| `--language, -l` | str | `ja` | 言語コード（ja/en/fr/de等） |
| `--min-confidence` | float | `0.3` | 最小信頼度閾値（0.0-1.0） |
| `--chunk-size` | int | `15` | 音声チャンクサイズ（秒） |
| `--use-advanced-vad` | flag | `True` | 高度VAD使用 |
| `--realtime-mode` | flag | `False` | リアルタイム処理モード |
| `--no-enhanced-preprocessing` | flag | `False` | 高度前処理をスキップ |
| `--no-post-processing` | flag | `False` | 後処理をスキップ |
| `--no-speaker-diarization` | flag | `False` | 話者識別をスキップ |
| `--hf-token` | str | `None` | HuggingFaceトークン |
| `--format` | str | `all` | 出力形式（all/json/csv/txt/srt） |
| `--device` | str | `cpu` | 処理デバイス（cpu/cuda） |
| `--auto-confirm` | flag | `False` | 確認ダイアログをスキップ |

#### 使用例

```bash
# 基本使用
python -m transcription.turbo_enhanced_main audio.mp3

# 高速処理（リアルタイムモード）
python -m transcription.turbo_enhanced_main audio.mp3 --realtime-mode --no-speaker-diarization

# カスタマイズ例
python -m transcription.turbo_enhanced_main audio.mp3 \
  --model base \
  --chunk-size 10 \
  --min-confidence 0.5 \
  --format json \
  --output my_result
```

### 2. Maximum Precision モード

#### 基本構文
```bash
python -m transcription.maximum_precision_main [AUDIO_FILE] [OPTIONS]
```

#### 専用オプション

| オプション | 型 | デフォルト | 説明 |
|------------|-----|------------|------|
| `--use-ensemble` | flag | `False` | アンサンブル処理有効化 |
| `--ensemble-models` | str | `base,medium` | アンサンブル用モデルリスト |
| `--voting-method` | str | `confidence_weighted` | 投票方式（confidence_weighted/majority） |
| `--use-advanced-vad` | flag | `True` | 高度VAD使用 |

#### 使用例

```bash
# 最高精度モード
python -m transcription.maximum_precision_main audio.mp3 --use-ensemble

# カスタムアンサンブル
python -m transcription.maximum_precision_main audio.mp3 \
  --use-ensemble \
  --ensemble-models \"medium,large-v3\" \
  --voting-method majority
```

### 3. 戻り値

全てのコマンドライン実行は以下のような出力を生成：

```bash
✅ 転写完了！
📊 結果サマリー:
   - 平均信頼度: 80.5%
   - 総セグメント: 86
   - 処理時間: 1.2分
   - 速度倍率: 8.1x

📁 保存ファイル:
   - JSON: audio_turbo.json
   - CSV: audio_turbo.csv  
   - TXT: audio_turbo.txt
   - SRT: audio_turbo.srt
```

## Python API

### 1. 高レベル API

#### turbo_enhanced_transcribe()

```python
def turbo_enhanced_transcribe(
    audio_file: str,
    output: Optional[str] = None,
    model: str = \"large-v3\",
    turbo_mode: bool = True,
    language: str = \"ja\",
    min_confidence: float = 0.3,
    chunk_size: int = 15,
    use_advanced_vad: bool = True,
    realtime_mode: bool = False,
    no_enhanced_preprocessing: bool = False,
    no_post_processing: bool = False,
    no_speaker_diarization: bool = False,
    hf_token: Optional[str] = None,
    output_format: str = \"all\",
    device: str = \"cpu\",
    auto_confirm: bool = False
) -> Dict[str, Any]:
    \"\"\"
    Turbo強化音声文字起こし
    
    Args:
        audio_file: 音声ファイルパス
        output: 出力ファイルベース名
        model: Whisperモデルサイズ
        turbo_mode: Turbo最適化有効化
        language: 言語コード
        min_confidence: 最小信頼度閾値
        chunk_size: チャンクサイズ（秒）
        use_advanced_vad: 高度VAD使用
        realtime_mode: リアルタイムモード
        no_enhanced_preprocessing: 前処理スキップ
        no_post_processing: 後処理スキップ
        no_speaker_diarization: 話者識別スキップ
        hf_token: HuggingFaceトークン
        output_format: 出力形式
        device: 処理デバイス
        auto_confirm: 自動確認
        
    Returns:
        Dict[str, Any]: 処理結果とメタデータ
        
    Raises:
        FileNotFoundError: 音声ファイルが見つからない
        ValueError: 無効なパラメータ
        RuntimeError: 処理エラー
    \"\"\"
```

#### 使用例

```python
from transcription.turbo_enhanced_main import turbo_enhanced_transcribe

# 基本使用
result = turbo_enhanced_transcribe(\"audio.mp3\")

# カスタマイズ
result = turbo_enhanced_transcribe(
    audio_file=\"audio.mp3\",
    model=\"base\",
    turbo_mode=True,
    realtime_mode=True,
    language=\"ja\",
    min_confidence=0.5,
    auto_confirm=True
)

print(f\"平均信頼度: {result['metadata']['average_confidence']:.1%}\")
print(f\"処理時間: {result['metadata']['processing_time_seconds']:.1f}秒\")
```

### 2. 中レベル API

#### EnhancedAudioProcessor

```python
class EnhancedAudioProcessor:
    def __init__(self, sample_rate: int = 16000):
        \"\"\"
        音声処理器初期化
        
        Args:
            sample_rate: サンプリングレート
        \"\"\"
    
    def advanced_preprocess_audio(
        self,
        file_path: str,
        enable_noise_reduction: bool = True,
        enable_speech_enhancement: bool = True,
        enable_spectral_norm: bool = True,
        enable_volume_adjustment: bool = True,
        enable_silence_removal: bool = False,
        memory_efficient: bool = False
    ) -> Tuple[np.ndarray, int]:
        \"\"\"
        高度音声前処理
        
        Args:
            file_path: 音声ファイルパス
            enable_noise_reduction: ノイズ除去有効化
            enable_speech_enhancement: 音声強化有効化  
            enable_spectral_norm: スペクトル正規化有効化
            enable_volume_adjustment: 音量調整有効化
            enable_silence_removal: 無音除去有効化
            memory_efficient: メモリ効率モード
            
        Returns:
            Tuple[np.ndarray, int]: (音声データ, サンプリングレート)
            
        Raises:
            FileNotFoundError: ファイルが見つからない
            ValueError: 無効な音声形式
        \"\"\"
```

#### FasterTranscriber

```python
class FasterTranscriber:
    def __init__(
        self,
        model_size: str = \"medium\",
        language: str = \"ja\",
        device: str = \"cpu\"
    ):
        \"\"\"
        高速転写器初期化
        
        Args:
            model_size: モデルサイズ
            language: 言語コード
            device: 処理デバイス
        \"\"\"
    
    def process_transcription(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        filter_confidence: bool = True,
        filter_fillers: bool = True,
        min_confidence: float = 0.3
    ) -> List[TranscriptionSegment]:
        \"\"\"
        音声転写処理
        
        Args:
            audio_data: 音声データ
            sample_rate: サンプリングレート
            filter_confidence: 信頼度フィルタリング
            filter_fillers: フィラーワード除去
            min_confidence: 最小信頼度
            
        Returns:
            List[TranscriptionSegment]: 転写結果
        \"\"\"
```

#### 使用例

```python
from transcription.enhanced_audio_processor import EnhancedAudioProcessor
from transcription.faster_transcriber import FasterTranscriber

# 音声前処理
processor = EnhancedAudioProcessor()
audio_data, sample_rate = processor.advanced_preprocess_audio(
    \"audio.mp3\",
    enable_noise_reduction=True,
    memory_efficient=True
)

# 転写処理
transcriber = FasterTranscriber(model_size=\"base\", device=\"cpu\")
segments = transcriber.process_transcription(
    audio_data,
    sample_rate,
    filter_confidence=True,
    min_confidence=0.5
)

# 結果処理
for segment in segments:
    print(f\"[{segment.start_time:.1f}-{segment.end_time:.1f}] {segment.text}\")
```

### 3. 低レベル API

#### EnsembleTranscriber

```python
class EnsembleTranscriber:
    def __init__(self, models: List[str], device: str = \"cpu\"):
        \"\"\"アンサンブル転写器初期化\"\"\"
    
    def ensemble_transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        voting_method: str = \"confidence_weighted\",
        quality_check: bool = True
    ) -> Tuple[List[dict], Dict[str, Any]]:
        \"\"\"
        アンサンブル転写実行
        
        Args:
            audio_data: 音声データ
            sample_rate: サンプリングレート
            voting_method: 投票方式
            quality_check: 品質チェック有効化
            
        Returns:
            Tuple[List[dict], Dict[str, Any]]: (転写結果, パフォーマンス情報)
        \"\"\"
```

## データ形式

### 1. TranscriptionSegment

```python
@dataclass
class TranscriptionSegment:
    start_time: float       # 開始時刻（秒）
    end_time: float         # 終了時刻（秒）
    text: str              # 転写テキスト
    confidence: float       # 信頼度（0.0-1.0）
    speaker_id: Optional[str] = None  # 話者ID
    
    def to_dict(self) -> dict:
        \"\"\"辞書形式に変換\"\"\"
        
    def format_timestamp(self, seconds: float) -> str:
        \"\"\"タイムスタンプフォーマット（HH:MM:SS.mmm）\"\"\"
```

### 2. JSON出力形式

```json
{
  \"metadata\": {
    \"input_file\": \"audio.mp3\",
    \"model_size\": \"base\",
    \"language\": \"ja\",
    \"device\": \"cpu\",
    \"engine\": \"turbo-enhanced\",
    \"turbo_mode\": true,
    \"realtime_mode\": false,
    \"chunk_size_seconds\": 15,
    \"techniques_applied\": {
      \"turbo_optimization\": true,
      \"advanced_vad\": true,
      \"enhanced_preprocessing\": true,
      \"post_processing\": true,
      \"speaker_diarization\": false,
      \"chunked_processing\": false,
      \"realtime_optimization\": false
    },
    \"min_confidence\": 0.3,
    \"audio_duration_seconds\": 600.0,
    \"total_segments\": 86,
    \"average_confidence\": 0.805,
    \"total_text_length\": 2613,
    \"processing_time_seconds\": 74.35,
    \"processing_ratio\": 0.124,
    \"actual_speedup\": 8.07,
    \"file_size_mb\": 54.9
  },
  \"segments\": [
    {
      \"segment_id\": 1,
      \"start_time\": \"00:00:00.000\",
      \"end_time\": \"00:00:06.520\",
      \"start_seconds\": 0.0,
      \"end_seconds\": 6.52,
      \"duration\": 6.52,
      \"speaker_id\": \"SPEAKER_UNKNOWN\",
      \"text\": \"こんにちは、今日は会議を始めさせていただきます。\",
      \"confidence\": 0.798
    }
  ],
  \"generated_at\": \"2025-06-01T12:12:06.871452\"
}
```

### 3. CSV出力形式

| カラム | 型 | 説明 |
|--------|-----|------|
| `segment_id` | int | セグメント連番 |
| `start_time` | str | 開始時刻（HH:MM:SS.mmm） |
| `end_time` | str | 終了時刻（HH:MM:SS.mmm） |
| `start_seconds` | float | 開始時刻（秒） |
| `end_seconds` | float | 終了時刻（秒） |
| `duration` | float | 継続時間（秒） |
| `speaker_id` | str | 話者ID |
| `text` | str | 転写テキスト |
| `confidence` | float | 信頼度 |

### 4. SRT出力形式

```srt
1
00:00:00,000 --> 00:00:06,520
[SPEAKER_UNKNOWN] こんにちは、今日は会議を始めさせていただきます。

2
00:00:06,520 --> 00:00:12,340
[SPEAKER_UNKNOWN] 議題は来月のプロジェクトについてです。
```

## エラーハンドリング

### 1. 一般的なエラー

#### FileNotFoundError
```python
# 原因: 指定された音声ファイルが存在しない
# 解決: ファイルパスを確認

try:
    result = turbo_enhanced_transcribe(\"nonexistent.mp3\")
except FileNotFoundError as e:
    print(f\"ファイルが見つかりません: {e}\")
```

#### ValueError
```python
# 原因: 無効なパラメータ値
# 解決: パラメータ値を修正

try:
    result = turbo_enhanced_transcribe(\"audio.mp3\", min_confidence=1.5)  # 無効値
except ValueError as e:
    print(f\"無効なパラメータ: {e}\")
```

#### RuntimeError
```python
# 原因: Whisperモデル読み込みエラー、CUDA関連エラー等
# 解決: モデル再ダウンロード、CPUモード切り替え

try:
    result = turbo_enhanced_transcribe(\"audio.mp3\", device=\"cuda\")
except RuntimeError as e:
    print(f\"処理エラー: {e}\")
    # CPUモードで再試行
    result = turbo_enhanced_transcribe(\"audio.mp3\", device=\"cpu\")
```

### 2. 特定エラー

#### HuggingFace認証エラー
```python
# pyannote.audioモデル利用時
try:
    result = turbo_enhanced_transcribe(\"audio.mp3\", hf_token=\"invalid_token\")
except Exception as e:
    if \"authentication\" in str(e).lower():
        print(\"HuggingFaceトークンが無効です\")
        # 話者識別をスキップして再実行
        result = turbo_enhanced_transcribe(\"audio.mp3\", no_speaker_diarization=True)
```

#### メモリ不足エラー
```python
# 大容量ファイル処理時
try:
    result = turbo_enhanced_transcribe(\"large_audio.mp3\")
except MemoryError:
    print(\"メモリ不足です。チャンク処理モードで再実行します\")
    # チャンクサイズを小さくして再実行
    result = turbo_enhanced_transcribe(\"large_audio.mp3\", chunk_size=5)
```

## 設定オプション

### 1. システム設定ファイル

設定ファイル（`configs/system_configs.yaml`）からデフォルト値を変更可能：

```yaml
# デフォルトモデル設定
models:
  default_model: \"medium\"
  turbo_model: \"base\"

# 音声処理設定
audio:
  sample_rate: 16000
  chunk_sizes:
    turbo_optimal: [10, 15]

# パフォーマンス設定
performance:
  max_memory_mb: 1024
  turbo_max_memory_mb: 512
```

### 2. 環境変数

| 環境変数 | 説明 | デフォルト |
|----------|------|------------|
| `HF_TOKEN` | HuggingFaceトークン | None |
| `CUDA_VISIBLE_DEVICES` | 使用GPU指定 | All |
| `WHISPER_CACHE_DIR` | モデルキャッシュディレクトリ | ~/.cache/whisper |

### 3. ランタイム設定

```python
# プログラム内での設定変更
import transcription.config as config

config.DEFAULT_MODEL = \"large-v3\"
config.DEFAULT_CHUNK_SIZE = 20
config.ENABLE_GPU = False
```

---

このAPIリファレンスを参照することで、音声文字起こしシステムを効果的に活用できます。詳細な実装例は`docs/DEVELOPER_GUIDE.md`をご覧ください。