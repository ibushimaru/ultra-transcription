# 🚀 Ultra Audio Transcription - Windows インストールガイド

## 📋 必要要件

### システム要件
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.8以上
- **メモリ**: 8GB以上（16GB推奨）
- **GPU**: NVIDIA GPU（RTX 2070 SUPER以上推奨）※CPUでも動作可能
- **ストレージ**: 10GB以上の空き容量

### 事前準備
1. **Python 3.8+** のインストール
   - [Python公式サイト](https://www.python.org/downloads/)からダウンロード
   - インストール時に「Add Python to PATH」にチェック

2. **NVIDIA GPU使用時のみ**
   - NVIDIA ドライバー（最新版）
   - CUDA Toolkit 12.1（自動インストールされます）

## 🛠️ インストール手順

### 1. ファイルのダウンロード
```bash
# GitHubからダウンロード（Gitがインストールされている場合）
git clone https://github.com/ultra-transcription/ultra-audio-transcription.git
cd ultra-audio-transcription
```

または、[リリースページ](https://github.com/ultra-transcription/ultra-audio-transcription/releases)からZIPファイルをダウンロードして解凍

### 2. 自動セットアップ
```bash
# セットアップスクリプトを実行
setup_windows.bat
```

このスクリプトは以下を自動的に行います：
- Python環境の確認
- 仮想環境の作成
- 必要なパッケージのインストール
- Whisper Turboモデルのダウンロード

### 3. インストール確認
```bash
# 仮想環境を有効化
venv\Scripts\activate.bat

# バージョン確認
ultra-transcribe --version
```

## 🎯 使い方

### 基本的な使用方法
```bash
# 音声ファイルを文字起こし（フィラーワード保持）
ultra-transcribe interview.mp3 -o interview_result

# フィラーワードを除外
ultra-transcribe interview.mp3 -o interview_clean --no-fillers

# 話者認識を無効化（高速処理）
ultra-transcribe meeting.wav -o meeting_fast --no-speaker
```

### バッチファイルを使用
`ultra-transcribe.bat`をダブルクリックして使用することもできます：
```bash
ultra-transcribe.bat audio.mp3 -o output
```

### 出力ファイル
以下の形式で出力されます：
- `output_ultra_precision.json` - 詳細な文字起こし結果
- `output_ultra_precision.csv` - 表形式の結果
- `output_ultra_precision.srt` - 字幕ファイル

## 🔧 トラブルシューティング

### Python が見つからない
```
ERROR: Python is not installed or not in PATH
```
→ Pythonを再インストールし、PATHに追加してください

### CUDA エラー
```
CUDA out of memory
```
→ GPUメモリが不足しています。CPUモードで実行：
```bash
ultra-transcribe audio.mp3 -o output --device cpu
```

### モデルダウンロードエラー
初回実行時はモデルのダウンロードに時間がかかります（約2GB）。
ネットワーク接続を確認してください。

## 🚀 高度な使用方法

### 長時間ファイルの処理
```bash
# 1時間以上のファイルには大きめのチャンクサイズ
ultra-transcribe long_audio.mp3 -o long_result --chunk-size 10
```

### 言語の指定
```bash
# 英語の音声
ultra-transcribe english.mp3 -o english_result --language en
```

### GPU/CPUの選択
```bash
# GPU使用（デフォルト）
ultra-transcribe audio.mp3 -o output

# CPU使用
ultra-transcribe audio.mp3 -o output --device cpu
```

## 📊 性能目安

| 音声長さ | GPU (RTX 2070S) | CPU (i7-9700K) |
|---------|----------------|----------------|
| 30秒    | 2-3秒          | 15-20秒        |
| 5分     | 30-40秒        | 3-4分          |
| 1時間   | 7-10分         | 45-60分        |

## 📞 サポート

問題が発生した場合：
1. [トラブルシューティングガイド](docs/TROUBLESHOOTING.md)を確認
2. [GitHubのIssues](https://github.com/ultra-transcription/ultra-audio-transcription/issues)で報告
3. [ディスカッション](https://github.com/ultra-transcription/ultra-audio-transcription/discussions)で質問

## 🎉 アップデート

最新版へのアップデート：
```bash
# 仮想環境を有効化
venv\Scripts\activate.bat

# 最新版をプル（Gitを使用している場合）
git pull

# パッケージを更新
pip install -e . --upgrade
```

---

**Ultra Audio Transcription v3.0.0** - Powered by Whisper Large-v3 Turbo 🚀