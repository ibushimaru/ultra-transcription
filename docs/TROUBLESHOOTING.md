# トラブルシューティングガイド

## 目次
1. [インストール関連の問題](#インストール関連の問題)
2. [実行時エラー](#実行時エラー)
3. [パフォーマンスの問題](#パフォーマンスの問題)
4. [出力品質の問題](#出力品質の問題)
5. [システム要件の問題](#システム要件の問題)
6. [設定の問題](#設定の問題)
7. [デバッグ方法](#デバッグ方法)

## インストール関連の問題

### 💽 依存関係インストールエラー

#### 問題: `pip install -r requirements.txt` が失敗する

**症状**:
```bash
ERROR: Could not build wheels for [package-name]
```

**解決方法**:

1. **Python バージョン確認**
```bash
python --version  # 3.8以上が必要
```

2. **pip アップデート**
```bash
pip install --upgrade pip setuptools wheel
```

3. **システム依存関係インストール（Ubuntu/Debian）**
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install ffmpeg
```

4. **システム依存関係インストール（macOS）**
```bash
brew install portaudio ffmpeg
```

5. **conda環境使用（推奨）**
```bash
conda create -n transcription python=3.9
conda activate transcription
pip install -r requirements.txt
```

#### 問題: torch/PyTorchインストールエラー

**解決方法**:
```bash
# CPU版（軽量）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版（CUDA 11.8）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 🔧 ライブラリ固有のエラー

#### 問題: `faster-whisper` インストールエラー

**解決方法**:
```bash
# 直接インストール
pip install faster-whisper

# バージョン指定
pip install faster-whisper==0.9.0
```

#### 問題: `pyannote.audio` エラー

**解決方法**:
```bash
# HuggingFace認証設定
pip install transformers[torch]
huggingface-cli login  # トークン入力

# または話者識別を無効化
python -m transcription.turbo_enhanced_main audio.mp3 --no-speaker-diarization
```

## 実行時エラー

### ❌ モジュール/ファイル関連エラー

#### 問題: `ModuleNotFoundError: No module named 'transcription'`

**原因**: パッケージが正しくインストールされていない

**解決方法**:
```bash
# 1. プロジェクトディレクトリで実行
cd /path/to/transcription

# 2. Python パス確認
python -c "import sys; print(sys.path)"

# 3. 開発モードでインストール
pip install -e .

# 4. 直接実行
python transcription/turbo_enhanced_main.py audio.mp3
```

#### 問題: `FileNotFoundError: Audio file not found`

**原因**: 指定された音声ファイルが存在しない

**解決方法**:
```bash
# 1. ファイル存在確認
ls -la audio.mp3

# 2. 絶対パス使用
python -m transcription.turbo_enhanced_main /full/path/to/audio.mp3

# 3. ファイル形式確認
file audio.mp3  # 対応形式: MP3, WAV, M4A, AAC, FLAC
```

### 🧠 メモリ関連エラー

#### 問題: `CUDA out of memory`

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XXMiB
```

**解決方法**:

1. **CPUモードに切り替え**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --device cpu
```

2. **小さいモデル使用**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --model base --device cuda
```

3. **チャンクサイズ削減**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --chunk-size 5 --device cuda
```

4. **GPU メモリクリア**
```python
import torch
torch.cuda.empty_cache()
```

#### 問題: `MemoryError` または `Out of memory`

**解決方法**:

1. **Turbo リアルタイムモード使用**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --realtime-mode
```

2. **前処理を最小化**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --no-enhanced-preprocessing
```

3. **話者識別を無効化**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --no-speaker-diarization
```

### 🔐 認証エラー

#### 問題: HuggingFace認証エラー

**症状**:
```
Could not download 'pyannote/speaker-diarization-3.1' pipeline.
```

**解決方法**:

1. **HuggingFaceトークン取得・設定**
```bash
# https://hf.co/settings/tokens でトークン生成
export HF_TOKEN=your_token_here
python -m transcription.turbo_enhanced_main audio.mp3 --hf-token $HF_TOKEN
```

2. **モデルアクセス許可**
- https://hf.co/pyannote/speaker-diarization-3.1 でUser Conditions承認

3. **話者識別スキップ**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --no-speaker-diarization
```

## パフォーマンスの問題

### 🐌 処理速度が遅い

#### 症状: 期待より大幅に処理が遅い

**診断・解決手順**:

1. **モード確認**
```bash
# 最高速度モード
python -m transcription.turbo_enhanced_main audio.mp3 --realtime-mode

# 速度重視設定
python -m transcription.turbo_enhanced_main audio.mp3 \
  --model base \
  --chunk-size 10 \
  --no-speaker-diarization \
  --no-enhanced-preprocessing
```

2. **GPU利用確認**
```bash
# GPU使用
python -m transcription.turbo_enhanced_main audio.mp3 --device cuda

# GPU状態確認
nvidia-smi  # NVIDIA GPU
```

3. **チャンクサイズ最適化**
```bash
# 短いファイル: 小さいチャンク
python -m transcription.turbo_enhanced_main short.mp3 --chunk-size 5

# 長いファイル: 大きいチャンク
python -m transcription.turbo_enhanced_main long.mp3 --chunk-size 20
```

#### 症状: 大容量ファイルで処理が停止

**解決方法**:

1. **自動チャンク処理確認**
```bash
# 30分以上のファイルは自動的にチャンク処理される
python -m transcription.turbo_enhanced_main large_file.mp3
```

2. **明示的にチャンク処理強制**
```python
from transcription.chunked_transcriber import ChunkedTranscriber

transcriber = ChunkedTranscriber(
    model_size=\"base\",
    chunk_duration=5,  # 5分チャンク
    max_memory_mb=512
)
result = transcriber.process_large_file(\"large_file.mp3\")
```

### 📊 メモリ使用量が多い

**解決方法**:

1. **メモリ効率モード有効化**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --realtime-mode
```

2. **モデルサイズ削減**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --model tiny
```

3. **バッチサイズ調整**
```python
# カスタム設定
config.MAX_MEMORY_MB = 512
config.CHUNK_SIZE = 5
```

## 出力品質の問題

### 📝 転写精度が低い

#### 症状: 信頼度が期待より低い（<70%）

**改善方法**:

1. **最高精度モード使用**
```bash
python -m transcription.maximum_precision_main audio.mp3 --use-ensemble
```

2. **音声品質確認**
```bash
# 音声ファイル情報確認
ffprobe audio.mp3

# ノイズ確認・改善
python -m transcription.turbo_enhanced_main audio.mp3 \
  --chunk-size 30 \
  --min-confidence 0.2
```

3. **言語設定確認**
```bash
# 日本語音声の場合
python -m transcription.turbo_enhanced_main audio.mp3 --language ja

# 英語音声の場合  
python -m transcription.turbo_enhanced_main audio.mp3 --language en
```

4. **信頼度閾値調整**
```bash
# より多くのセグメントを取得
python -m transcription.turbo_enhanced_main audio.mp3 --min-confidence 0.1
```

#### 症状: 話者識別が不正確

**解決方法**:

1. **HuggingFaceトークン設定確認**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --hf-token YOUR_TOKEN
```

2. **音声品質改善**
- 複数話者が明確に分離されている音声を使用
- ノイズの少ない環境での録音
- 十分な音量レベル

3. **手動話者ラベリング**
```python
# 時間ベースで手動割り当て
segments = transcription_result['segments']
for segment in segments:
    if segment['start_seconds'] < 300:  # 最初の5分
        segment['speaker_id'] = 'SPEAKER_A'
    else:
        segment['speaker_id'] = 'SPEAKER_B'
```

### 🎯 セグメント分割が不適切

**解決方法**:

1. **VAD設定調整**
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --use-advanced-vad
```

2. **チャンクサイズ調整**
```bash
# 短い発言が多い場合
python -m transcription.turbo_enhanced_main audio.mp3 --chunk-size 5

# 長い発言が多い場合
python -m transcription.turbo_enhanced_main audio.mp3 --chunk-size 30
```

3. **後処理無効化**
```bash
# セグメント統合を無効化
python -m transcription.turbo_enhanced_main audio.mp3 --no-post-processing
```

## システム要件の問題

### 💻 ハードウェア不足

#### 症状: 「システム要件を満たさない」エラー

**最小要件**:
- Python 3.8+
- RAM: 2GB以上
- ストレージ: 5GB以上
- CPU: 2コア以上

**推奨要件**:
- Python 3.9+
- RAM: 8GB以上
- ストレージ: 10GB以上
- CPU: 4コア以上
- GPU: NVIDIA GTX 1060以上（オプション）

**軽量化設定**:
```bash
# 最軽量設定
python -m transcription.turbo_enhanced_main audio.mp3 \
  --model tiny \
  --realtime-mode \
  --no-enhanced-preprocessing \
  --no-speaker-diarization \
  --chunk-size 5
```

### 🌐 ネットワーク問題

#### 症状: モデルダウンロードエラー

**解決方法**:

1. **プロキシ設定**
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
pip install --proxy http://proxy.company.com:8080 -r requirements.txt
```

2. **オフライン環境**
```bash
# 別環境でモデル事前ダウンロード
python -c \"import whisper; whisper.load_model('base')\"

# モデルファイルをコピー
cp ~/.cache/whisper/* /target/environment/.cache/whisper/
```

## 設定の問題

### ⚙️ 設定ファイルエラー

#### 問題: YAML設定ファイル読み込みエラー

**解決方法**:

1. **YAML構文確認**
```bash
python -c "import yaml; yaml.safe_load(open('configs/system_configs.yaml'))"
```

2. **デフォルト設定使用**
```bash
# 設定ファイル無視
python -m transcription.turbo_enhanced_main audio.mp3
```

3. **設定ファイル再生成**
```bash
cp configs/system_configs.yaml.example configs/system_configs.yaml
```

### 🔧 環境変数の問題

**確認方法**:
```bash
# 重要な環境変数確認
echo $HF_TOKEN
echo $CUDA_VISIBLE_DEVICES
echo $PYTHONPATH
```

**設定方法**:
```bash
# 一時設定
export HF_TOKEN=your_token

# 永続設定（~/.bashrc に追加）
echo 'export HF_TOKEN=your_token' >> ~/.bashrc
source ~/.bashrc
```

## デバッグ方法

### 🔍 詳細ログ出力

#### 基本デバッグ情報
```bash
# 詳細ログ有効化
python -m transcription.turbo_enhanced_main audio.mp3 --verbose

# Python デバッグモード
PYTHONPATH=. python -u transcription/turbo_enhanced_main.py audio.mp3
```

#### 段階的診断
```python
# 1. 音声ファイル読み込み確認
from transcription.enhanced_audio_processor import EnhancedAudioProcessor
processor = EnhancedAudioProcessor()
audio_data, sr = processor.load_audio(\"audio.mp3\")
print(f\"Audio loaded: {len(audio_data)} samples at {sr}Hz\")

# 2. モデル読み込み確認
from transcription.faster_transcriber import FasterTranscriber
transcriber = FasterTranscriber(model_size=\"base\")
print(\"Model loaded successfully\")

# 3. 小さいサンプルでテスト
sample = audio_data[:sr*10]  # 最初の10秒
segments = transcriber.process_transcription(sample, sr)
print(f\"Generated {len(segments)} segments\")
```

### 📋 システム情報収集

#### 診断スクリプト
```python
import sys
import torch
import platform
import psutil

def system_diagnosis():
    print(f\"Python: {sys.version}\")
    print(f\"Platform: {platform.platform()}\")
    print(f\"CPU cores: {psutil.cpu_count()}\")
    print(f\"Memory: {psutil.virtual_memory().total // (1024**3)}GB\")
    
    if torch.cuda.is_available():
        print(f\"CUDA: {torch.cuda.get_device_name()}\")
        print(f\"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB\")
    else:
        print(\"CUDA: Not available\")
    
    try:
        import whisper
        print(f\"Whisper: {whisper.__version__}\")
    except:
        print(\"Whisper: Not installed\")

system_diagnosis()
```

### 🚨 緊急時対処法

#### 完全リセット手順
```bash
# 1. 仮想環境削除・再作成
conda deactivate
conda env remove -n transcription
conda create -n transcription python=3.9
conda activate transcription

# 2. 依存関係再インストール
pip install -r requirements.txt

# 3. キャッシュクリア
rm -rf ~/.cache/whisper/
rm -rf ~/.cache/huggingface/

# 4. 最小構成でテスト
python -m transcription.turbo_enhanced_main test.mp3 \
  --model tiny \
  --no-speaker-diarization \
  --realtime-mode
```

## サポート・報告

### 🆘 問題が解決しない場合

1. **GitHub Issue作成**: [リポジトリURL]/issues
2. **必要情報**:
   - エラーメッセージ全文
   - 使用コマンド
   - システム情報（上記診断スクリプト結果）
   - 音声ファイル情報（可能な範囲で）

3. **ログファイル添付**:
```bash
python -m transcription.turbo_enhanced_main audio.mp3 --verbose > debug.log 2>&1
```

---

📧 **緊急サポート**: 本番環境での重大な問題は、詳細な環境情報とともにIssueを報告してください。
📚 **詳細情報**: 技術仕様は`docs/DEVELOPER_GUIDE.md`、API仕様は`docs/API_REFERENCE.md`をご覧ください。