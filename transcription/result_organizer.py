#!/usr/bin/env python3
"""
Result Organizer - テスト結果整理システム

テスト結果を体系的に整理し、メタデータを管理するシステム:
- モデル別フォルダ分類
- テスト種別分類  
- メタデータ自動生成
- 比較分析サポート
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

class ResultOrganizer:
    """テスト結果整理システム"""
    
    def __init__(self, base_dir: str = "test_outputs"):
        """
        Initialize result organizer
        
        Args:
            base_dir: ベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.organized_dir = self.base_dir / "organized"
        
        # ログ設定
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 新フォルダ構造
        self.structure = {
            "benchmarks": {
                "short_tests": "30秒以下のテスト",
                "medium_tests": "1-5分のテスト", 
                "long_tests": "大容量ファイルテスト"
            },
            "models": {
                "tiny": "tiny モデル結果",
                "small": "small モデル結果",
                "medium": "medium モデル結果", 
                "large": "large モデル結果",
                "large-v3-turbo": "large-v3-turbo モデル結果",
                "turbo": "turbo モデル結果"
            },
            "systems": {
                "rapid_ultra": "rapid_ultra_processor結果",
                "large_file_ultra": "large_file_ultra_precision結果",
                "segmented": "segmented_processor結果",
                "ultra_precision": "ultra_precision_speaker結果"
            },
            "reference": {
                "quality_baseline": "品質参照ベースライン",
                "performance_baseline": "性能参照ベースライン"
            },
            "production": {
                "final_results": "本番使用可能な最終結果"
            }
        }
    
    def create_organized_structure(self):
        """新しいフォルダ構造を作成"""
        self.logger.info("Creating organized directory structure...")
        
        for category, subcategories in self.structure.items():
            category_path = self.organized_dir / category
            category_path.mkdir(parents=True, exist_ok=True)
            
            if isinstance(subcategories, dict):
                for subcat, description in subcategories.items():
                    subcat_path = category_path / subcat
                    subcat_path.mkdir(exist_ok=True)
                    
                    # READMEファイル作成
                    readme_path = subcat_path / "README.md"
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(f"# {subcat}\n\n{description}\n\n")
                        f.write(f"作成日: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def extract_metadata_from_json(self, json_file: Path) -> Dict[str, Any]:
        """JSONファイルからメタデータを抽出"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = {
                "file_name": json_file.name,
                "file_size_mb": json_file.stat().st_size / (1024 * 1024),
                "created_date": datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(),
                "segments_count": len(data.get('segments', [])),
                "format_version": data.get('format_version', 'unknown')
            }
            
            # メタデータ section があれば追加
            if 'metadata' in data:
                meta = data['metadata']
                metadata.update({
                    "model_used": meta.get('model_used', 'unknown'),
                    "language": meta.get('language', 'unknown'),
                    "processing_method": meta.get('processing_method', 'unknown'),
                    "total_duration": meta.get('total_duration', 0),
                    "average_confidence": meta.get('average_confidence', 0),
                    "processing_time": meta.get('processing_time', 0),
                    "real_time_factor": meta.get('real_time_factor', 0)
                })
            
            # summary section があれば追加
            if 'summary' in data:
                summary = data['summary']
                metadata.update({
                    "speaker_count": summary.get('speaker_count', 0),
                    "total_duration": summary.get('total_duration_seconds', 0)
                })
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {json_file}: {e}")
            return {
                "file_name": json_file.name,
                "error": str(e),
                "file_size_mb": json_file.stat().st_size / (1024 * 1024),
                "created_date": datetime.fromtimestamp(json_file.stat().st_mtime).isoformat()
            }
    
    def classify_file(self, file_path: Path) -> Dict[str, str]:
        """ファイルを分類"""
        file_name = file_path.name.lower()
        
        # モデル分類
        model = "unknown"
        if any(m in file_name for m in ["large-v3-turbo", "large_v3_turbo"]):
            model = "large-v3-turbo"
        elif "turbo" in file_name:
            model = "turbo"
        elif "large" in file_name:
            model = "large"
        elif "medium" in file_name:
            model = "medium"
        elif "small" in file_name:
            model = "small"
        elif "tiny" in file_name:
            model = "tiny"
        
        # システム分類
        system = "unknown"
        if "segmented" in file_name:
            system = "segmented"
        elif "rapid_ultra" in file_name or "rapid" in file_name:
            system = "rapid_ultra"
        elif "large_file" in file_name:
            system = "large_file_ultra"
        elif "ultra_precision" in file_name:
            system = "ultra_precision"
        
        # テストサイズ分類
        test_size = "unknown"
        if "ultra_short" in file_name or "30s" in file_name:
            test_size = "short"
        elif "90s" in file_name or "test_90" in file_name:
            test_size = "medium"
        elif "large_file" in file_name or "segmented" in file_name:
            test_size = "long"
        
        # 品質分類
        quality_type = "normal"
        if "final" in file_name or "optimized" in file_name:
            quality_type = "reference"
        elif "precision" in file_name:
            quality_type = "high_quality"
        
        return {
            "model": model,
            "system": system,
            "test_size": test_size,
            "quality_type": quality_type
        }
    
    def organize_files(self):
        """ファイルを整理"""
        self.logger.info("Starting file organization...")
        
        # 新構造作成
        self.create_organized_structure()
        
        # JSONファイルを取得
        json_files = list(self.base_dir.glob("*.json"))
        
        organized_files = []
        metadata_collection = []
        
        for json_file in json_files:
            # ファイル分類
            classification = self.classify_file(json_file)
            
            # メタデータ抽出
            metadata = self.extract_metadata_from_json(json_file)
            metadata.update(classification)
            metadata_collection.append(metadata)
            
            # 対応する他のファイル (.csv, .srt) も取得
            base_name = json_file.stem
            related_files = []
            
            for ext in ['.json', '.csv', '.srt']:
                related_file = json_file.with_suffix(ext)
                if related_file.exists():
                    related_files.append(related_file)
            
            # 保存先決定
            target_dir = self._determine_target_directory(classification)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # ファイル移動
            for file_path in related_files:
                target_file = target_dir / file_path.name
                if not target_file.exists():
                    shutil.copy2(file_path, target_file)
                    self.logger.info(f"Moved: {file_path.name} -> {target_dir}")
            
            organized_files.append({
                "original_path": str(json_file),
                "target_directory": str(target_dir),
                "classification": classification,
                "related_files": [f.name for f in related_files]
            })
        
        # メタデータサマリー作成
        self._create_metadata_summary(metadata_collection, organized_files)
        
        self.logger.info(f"Organization completed: {len(organized_files)} file sets processed")
        return organized_files
    
    def _determine_target_directory(self, classification: Dict[str, str]) -> Path:
        """分類に基づいて保存先ディレクトリを決定"""
        model = classification['model']
        system = classification['system']
        test_size = classification['test_size']
        quality_type = classification['quality_type']
        
        # 参照品質の場合
        if quality_type == "reference":
            return self.organized_dir / "reference" / "quality_baseline"
        
        # システム別分類を優先
        if system != "unknown":
            system_dir = self.organized_dir / "systems" / system
            if model != "unknown":
                return system_dir / model
            return system_dir
        
        # モデル別分類
        if model != "unknown":
            model_dir = self.organized_dir / "models" / model
            if test_size == "short":
                return model_dir / "short_tests"
            elif test_size == "medium":
                return model_dir / "medium_tests"
            elif test_size == "long":
                return model_dir / "long_tests"
            return model_dir
        
        # ベンチマーク分類
        if test_size == "short":
            return self.organized_dir / "benchmarks" / "short_tests"
        elif test_size == "medium":
            return self.organized_dir / "benchmarks" / "medium_tests"
        elif test_size == "long":
            return self.organized_dir / "benchmarks" / "long_tests"
        
        # デフォルト
        return self.organized_dir / "uncategorized"
    
    def _create_metadata_summary(self, metadata_collection: List[Dict], organized_files: List[Dict]):
        """メタデータサマリーを作成"""
        summary = {
            "organization_date": datetime.now().isoformat(),
            "total_files_processed": len(metadata_collection),
            "statistics": {
                "models": {},
                "systems": {},
                "test_sizes": {},
                "quality_types": {}
            },
            "files": metadata_collection,
            "organization_log": organized_files
        }
        
        # 統計計算
        for metadata in metadata_collection:
            for key in ["model", "system", "test_size", "quality_type"]:
                if key in metadata:
                    value = metadata[key]
                    if value not in summary["statistics"][f"{key}s"]:
                        summary["statistics"][f"{key}s"][value] = 0
                    summary["statistics"][f"{key}s"][value] += 1
        
        # サマリー保存
        summary_file = self.organized_dir / "metadata_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 人間が読みやすいサマリー作成
        readme_file = self.organized_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# テスト結果整理サマリー\n\n")
            f.write(f"整理日時: {summary['organization_date']}\n")
            f.write(f"処理ファイル数: {summary['total_files_processed']}\n\n")
            
            f.write("## 統計\n\n")
            for category, stats in summary["statistics"].items():
                f.write(f"### {category.title()}\n")
                for item, count in stats.items():
                    f.write(f"- {item}: {count}個\n")
                f.write("\n")
            
            f.write("## フォルダ構造\n\n")
            for category in self.structure.keys():
                f.write(f"- `{category}/`: {category}関連ファイル\n")
        
        self.logger.info(f"Metadata summary saved: {summary_file}")

def main():
    """コマンドラインインターフェース"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Result Organizer - テスト結果整理システム"
    )
    
    parser.add_argument('--base-dir', default='test_outputs',
                       help='ベースディレクトリ')
    parser.add_argument('--dry-run', action='store_true',
                       help='実際にファイルを移動せずに分析のみ')
    
    args = parser.parse_args()
    
    organizer = ResultOrganizer(args.base_dir)
    
    if args.dry_run:
        print("DRY RUN: ファイル分析のみ実行...")
        # TODO: dry run implementation
    else:
        organized_files = organizer.organize_files()
        print(f"\n✅ 整理完了: {len(organized_files)} ファイルセットを処理")
        print(f"📁 整理結果: {organizer.organized_dir}")

if __name__ == '__main__':
    main()