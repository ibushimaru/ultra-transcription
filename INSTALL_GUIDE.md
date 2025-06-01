# 📥 Ultra Audio Transcription - インストールガイド

## 🖥️ Windows ユーザー向け

### 推奨インストール方法

1. **GitHubからダウンロード**
   - [最新リリース](https://github.com/ibushimaru/ultra-transcription/releases/latest)からZIPファイルをダウンロード
   - または `git clone https://github.com/ibushimaru/ultra-transcription.git`

2. **インストーラーを実行**
   ```
   setup_windows.bat をダブルクリック
   ```

3. **エラーが発生した場合**
   ```
   quick_install_windows.bat をダブルクリック
   ```

### インストールスクリプトの違い

| スクリプト | 用途 | 特徴 |
|-----------|------|------|
| **setup_windows.bat** | メインインストーラー | pyproject.tomlを使用した完全インストール |
| **quick_install_windows.bat** | 代替インストーラー | requirements.txt不要、直接パッケージインストール |
| **run_windows.bat** | 実行用 | インストール後の文字起こし実行用 |

### よくある問題と解決策

#### ❌ "pip._vendor.tomli._parser.TOMLDecodeError" エラー
→ **quick_install_windows.bat** を使用してください

#### ❌ "requirements.txt not found" エラー
→ v3.0.2以降では修正済み。最新版をダウンロードしてください

#### ❌ "Python is not installed" エラー
→ [Python 3.8+](https://www.python.org/)をインストールし、PATHに追加してください

## 🐧 Linux/Mac ユーザー向け

### 標準インストール
```bash
# リポジトリをクローン
git clone https://github.com/ibushimaru/ultra-transcription.git
cd ultra-transcription

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Linuxの場合
# source venv/bin/activate.fish  # fishシェルの場合

# インストール
pip install -e .[gpu]  # GPU版
# または
pip install -e .       # CPU版
```

## 📱 使い方

### Windows
```bash
# 方法1: バッチファイル経由
ultra-transcribe.bat audio.mp3 -o result

# 方法2: 直接実行
run_windows.bat audio.mp3 -o result

# 方法3: Python経由（venv有効化後）
python -m transcription.rapid_ultra_processor audio.mp3 -o result
```

### Linux/Mac
```bash
# venv有効化後
ultra-transcribe audio.mp3 -o result
```

## 🎯 オプション

- **フィラーワード保持**（デフォルト）: 自然な会話のまま
- **フィラーワード除外**: `--no-fillers`
- **話者認識無効化**（高速）: `--no-speaker`

## 📞 サポート

問題が発生した場合：
1. [トラブルシューティング](docs/TROUBLESHOOTING.md)を確認
2. [GitHub Issues](https://github.com/ibushimaru/ultra-transcription/issues)で報告
3. [最新リリース](https://github.com/ibushimaru/ultra-transcription/releases)を確認