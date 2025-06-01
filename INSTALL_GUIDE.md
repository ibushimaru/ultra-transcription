# 📥 Ultra Audio Transcription - インストールガイド

## 🖥️ Windows ユーザー向け

### 推奨インストール方法

1. **GitHubからダウンロード**
   - [最新リリース](https://github.com/ibushimaru/ultra-transcription/releases/latest)からZIPファイルをダウンロード
   - または `git clone https://github.com/ibushimaru/ultra-transcription.git`

2. **インストーラーを実行**
   ```
   install.bat をダブルクリック
   ```
   
   このインストーラーは自動的にエラーを検出し、最適な方法でインストールします。

### インストールスクリプト

| スクリプト | 用途 | 特徴 |
|-----------|------|------|
| **install.bat** | 統合インストーラー | 自動フォールバック機能付き、pyproject.toml/requirements.txt両対応 |
| **ultra-transcribe.bat** | 実行用 | インストール後の文字起こし実行用（自動生成） |

### よくある問題と解決策

#### ❌ "pip._vendor.tomli._parser.TOMLDecodeError" エラー
→ install.bat が自動的に代替方法で対処します

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

#### 仮想環境を意識せずに実行

```bash
# 方法1: コマンドプロンプト
ultra-transcribe audio.mp3 -o result

# 方法2: PowerShell（初回セットアップ後）
ultra-transcribe audio.mp3 -o result

# セットアップ前、または現在のディレクトリから
.\ultra-transcribe.ps1 audio.mp3 -o result

# 方法3: デスクトップショートカット作成
powershell -ExecutionPolicy Bypass .\create_shortcut.ps1
# → デスクトップのショートカットに音声ファイルをドラッグ＆ドロップ
```

#### グローバルに使用したい場合

```bash
# PATHに追加（一度だけ実行）
add_to_path.bat

# その後、どこからでも実行可能
cd C:\MyAudioFiles
ultra-transcribe interview.mp3 -o result
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