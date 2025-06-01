# 音声文字起こしシステム ユーザーマニュアル

## 目次
1. [システム概要](#システム概要)
2. [インストール手順](#インストール手順)
3. [基本的な使い方](#基本的な使い方)
4. [各モードの詳細](#各モードの詳細)
5. [出力形式について](#出力形式について)
6. [設定オプション](#設定オプション)
7. [トラブルシューティング](#トラブルシューティング)
8. [よくある質問](#よくある質問)

## システム概要

### 🎯 何ができるの？
- **音声ファイル**（MP3、WAV、M4A等）を**テキスト**に変換
- **話者識別**機能で誰が話しているかを判別
- **タイムスタンプ**付きで正確な発言時刻を記録
- **複数の出力形式**（JSON、CSV、TXT、SRT字幕）
- **完全ローカル処理**でプライバシー保護

### 🚀 3つのモードで用途に最適化
| モード | 精度 | 速度 | 適用場面 |
|--------|------|------|----------|
| **Maximum Precision** | 🟢 87.2% | ⚡ 0.8x | 会議録、字幕制作、重要な記録 |
| **Turbo Enhanced** | 🟡 80.5% | ⚡ 8.1x | 日常的な文字起こし、一般用途 |
| **Turbo Realtime** | 🟠 79.5% | ⚡ 7.4x | ライブ配信、リアルタイム処理 |

## インストール手順

### 1. 必要な環境
- Python 3.8以上
- メモリ: 2GB以上推奨
- ストレージ: 5GB以上（モデルファイル含む）

### 2. パッケージインストール
```bash
# リポジトリをクローン
git clone [リポジトリURL]
cd transcription

# 依存関係をインストール
pip install -r requirements.txt

# 動作確認
python -m transcription.turbo_enhanced_main --help
```

### 3. 話者識別機能（オプション）
話者識別を使用する場合は、HuggingFaceトークンが必要です：

1. [HuggingFace](https://huggingface.co/)でアカウント作成
2. [Settings > Tokens](https://hf.co/settings/tokens)でトークン生成
3. 使用時に`--hf-token YOUR_TOKEN`を指定

## 基本的な使い方

### 🚀 最も簡単な使い方（推奨）
```bash
# Turbo Enhanced モード（速度と精度のバランス重視）
python -m transcription.turbo_enhanced_main audio.mp3
```

### 📊 最高精度で処理したい場合
```bash
# Maximum Precision モード
python -m transcription.maximum_precision_main audio.mp3 --use-ensemble
```

### ⚡ 超高速処理が必要な場合
```bash
# Turbo Realtime モード
python -m transcription.turbo_enhanced_main audio.mp3 --realtime-mode
```

## 各モードの詳細

### 1. Maximum Precision モード

**用途**: 会議録、インタビュー、字幕制作など、精度が最重要な場面

```bash
# 基本実行
python -m transcription.maximum_precision_main audio.mp3

# 詳細オプション付き
python -m transcription.maximum_precision_main audio.mp3 \
  --use-ensemble \
  --ensemble-models "base,medium" \
  --voting-method "confidence_weighted" \
  --min-confidence 0.2 \
  --hf-token YOUR_TOKEN
```

**特徴**:
- 🎯 87.2%の高精度
- 🔄 複数モデルのアンサンブル学習
- 🧠 信頼度重み付け投票
- ⏱️ 処理時間: 音声の約0.8倍（10分→8分）

### 2. Turbo Enhanced モード

**用途**: 日常的な文字起こし、一般的なビジネス用途

```bash
# 基本実行（推奨設定）
python -m transcription.turbo_enhanced_main audio.mp3

# カスタマイズ例
python -m transcription.turbo_enhanced_main audio.mp3 \
  --model base \
  --chunk-size 15 \
  --min-confidence 0.3 \
  --hf-token YOUR_TOKEN
```

**特徴**:
- ⚡ 8.1倍の高速処理
- 🎯 80.5%の十分な精度
- 💾 メモリ効率最適化
- ⏱️ 処理時間: 音声の約0.12倍（10分→1.2分）

### 3. Turbo Realtime モード

**用途**: ライブ配信、リアルタイム字幕、大量バッチ処理

```bash
# リアルタイムモード
python -m transcription.turbo_enhanced_main audio.mp3 \
  --realtime-mode \
  --no-speaker-diarization \
  --chunk-size 10
```

**特徴**:
- ⚡ 7.4倍の超高速処理
- 🎯 79.5%の実用的精度
- 📱 最小限のメモリ使用
- ⏱️ 処理時間: 音声の約0.13倍（10分→1.4分）

## 出力形式について

### 📁 自動生成されるファイル

実行すると、以下の形式で自動保存されます：

```
audio_turbo.json    # API連携用、メタデータ完備
audio_turbo.csv     # 表計算ソフト用、分析用
audio_turbo.txt     # 読みやすい形式、プレビュー用
audio_turbo.srt     # 字幕ファイル、動画編集用
```

### 📊 出力形式の詳細

#### JSON形式
```json
{
  \"metadata\": {
    \"average_confidence\": 0.805,
    \"processing_time_seconds\": 74.35,
    \"total_segments\": 86
  },
  \"segments\": [
    {
      \"segment_id\": 1,
      \"start_time\": \"00:00:00.000\",
      \"end_time\": \"00:00:06.520\",
      \"speaker_id\": \"SPEAKER_01\",
      \"text\": \"こんにちは、今日は会議を始めさせていただきます。\",
      \"confidence\": 0.95
    }
  ]
}
```

#### CSV形式
| segment_id | start_time | end_time | speaker_id | text | confidence |
|------------|------------|----------|------------|------|------------|
| 1 | 00:00:00.000 | 00:00:06.520 | SPEAKER_01 | こんにちは... | 0.95 |

#### TXT形式
```
音声文字起こし結果
==================================================

[SPEAKER_01]
[00:00:00.000 - 00:00:06.520] (0.95) こんにちは、今日は会議を始めさせていただきます。
[00:00:06.520 - 00:00:12.340] (0.88) 議題は来月のプロジェクトについてです。
```

#### SRT形式（字幕）
```
1
00:00:00,000 --> 00:00:06,520
[SPEAKER_01] こんにちは、今日は会議を始めさせていただきます。

2
00:00:06,520 --> 00:00:12,340
[SPEAKER_01] 議題は来月のプロジェクトについてです。
```

## 設定オプション

### 🎛️ 主要オプション

| オプション | 説明 | デフォルト | 例 |
|------------|------|------------|-----|
| `--model` | Whisperモデルサイズ | medium | `--model base` |
| `--language` | 言語コード | ja | `--language en` |
| `--min-confidence` | 最小信頼度閾値 | 0.3 | `--min-confidence 0.5` |
| `--chunk-size` | チャンクサイズ（秒） | 15 | `--chunk-size 10` |
| `--output` | 出力ファイル名 | 自動生成 | `--output my_result` |
| `--format` | 出力形式 | all | `--format json` |

### 🔧 詳細オプション

```bash
# 前処理をスキップ（高速化）
--no-enhanced-preprocessing

# 後処理をスキップ
--no-post-processing

# 話者識別をスキップ
--no-speaker-diarization

# デバイス指定
--device cuda

# 自動確認（バッチ処理用）
--auto-confirm
```

## トラブルシューティング

### ❌ よくあるエラーと解決法

#### 1. 「ModuleNotFoundError: No module named 'faster_whisper'」
```bash
# 解決法: 依存関係を再インストール
pip install -r requirements.txt
```

#### 2. 「CUDA out of memory」
```bash
# 解決法: CPUモードで実行
python -m transcription.turbo_enhanced_main audio.mp3 --device cpu
```

#### 3. 「Speaker diarization failed」
```bash
# 解決法: 話者識別をスキップ
python -m transcription.turbo_enhanced_main audio.mp3 --no-speaker-diarization
```

#### 4. 処理が非常に遅い
```bash
# 解決法: Turbo Realtimeモードを使用
python -m transcription.turbo_enhanced_main audio.mp3 --realtime-mode
```

#### 5. 音声ファイルが読み込めない
対応形式を確認：MP3, WAV, M4A, AAC, FLAC

### 🐛 ログの確認方法

エラーが発生した場合は、詳細ログを確認：

```bash
# デバッグモードで実行
python -m transcription.turbo_enhanced_main audio.mp3 --verbose
```

## よくある質問

### Q1: どのモードを選べばいい？
**A**: 迷ったら**Turbo Enhanced**モードがおすすめ。速度と精度のバランスが最適です。

### Q2: 大容量ファイル（2時間以上）も処理できる？
**A**: はい。30分以上のファイルは自動的にチャンク処理に切り替わります。

### Q3: 英語の音声も処理できる？
**A**: はい。`--language en`を指定してください。

### Q4: 話者識別が必要ない場合は？
**A**: `--no-speaker-diarization`を指定すると高速化されます。

### Q5: 字幕ファイルだけ欲しい場合は？
**A**: `--format srt`を指定してください。

### Q6: 処理時間を事前に知りたい
**A**: 全モードで処理前に予想時間が表示されます。

### Q7: 精度を上げるには？
**A**: 
- Maximum Precisionモード使用
- 音質の良い音声ファイルを使用
- ノイズの少ない環境で録音

### Q8: バッチ処理するには？
**A**: `--auto-confirm`オプションでユーザー確認をスキップできます。

## 🎉 使用例

### ビジネス会議の議事録作成
```bash
python -m transcription.maximum_precision_main meeting.mp3 \
  --use-ensemble \
  --hf-token YOUR_TOKEN \
  --output meeting_minutes
```

### YouTube動画の字幕作成
```bash
python -m transcription.turbo_enhanced_main video_audio.wav \
  --format srt \
  --output video_subtitles
```

### ライブ配信のリアルタイム字幕
```bash
python -m transcription.turbo_enhanced_main stream.mp3 \
  --realtime-mode \
  --no-speaker-diarization \
  --format txt
```

### 英語インタビューの文字起こし
```bash
python -m transcription.maximum_precision_main interview.wav \
  --language en \
  --use-ensemble \
  --min-confidence 0.4
```

---

📧 **サポート**: 問題が解決しない場合は、GitHubのIssueでお知らせください。
📚 **詳細情報**: 開発者向けドキュメントやAPI仕様は`docs/`フォルダをご覧ください。