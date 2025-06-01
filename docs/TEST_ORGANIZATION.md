# 📂 Test Result Organization System

## Overview

The Test Result Organizer (`transcription/result_organizer.py`) provides automated organization and analysis of transcription test results, maintaining a structured hierarchy for easy access and comparison.

## 🎯 Purpose

- **Automated Organization**: Categorizes test files by model, system, and quality
- **Metadata Analysis**: Extracts and summarizes key metrics
- **Performance Tracking**: Enables easy comparison across tests
- **Clean Structure**: Maintains organized test output directory

## 📁 Directory Structure

```
test_outputs/organized/
├── models/                 # Model-specific tests
│   ├── large-v3/
│   │   ├── short_tests/   # <60s audio
│   │   ├── medium_tests/  # 60-600s audio
│   │   └── long_tests/    # >600s audio
│   ├── large-v3-turbo/
│   ├── large/
│   ├── medium/
│   ├── small/
│   └── tiny/
├── systems/               # Processing system tests
│   ├── ultra_precision/
│   ├── rapid_ultra/
│   ├── segmented/
│   └── large_file_ultra/
├── benchmarks/            # Performance benchmarks
│   ├── short_tests/
│   ├── medium_tests/
│   └── long_tests/
├── reference/             # Baseline reference results
│   ├── quality_baseline/
│   └── performance_baseline/
├── production/            # Production-ready results
│   └── final_results/
├── uncategorized/         # Files that couldn't be categorized
└── metadata_summary.json  # Organization summary
```

## 🔧 Usage

### Basic Usage
```bash
# Organize all test results
python3 -m transcription.result_organizer

# With custom paths
python3 -m transcription.result_organizer \
  --source test_outputs \
  --target test_outputs/organized
```

### Command-Line Options
- `--source`: Source directory containing test results (default: test_outputs)
- `--target`: Target directory for organized results (default: test_outputs/organized)
- `--dry-run`: Preview organization without moving files
- `--verbose`: Enable detailed logging

## 📊 Categorization Logic

### Model Detection
- Extracts from filename: `test_{model}_{duration}_{type}_{date}.json`
- Falls back to `model_used` field in JSON
- Maps variations (e.g., "turbo" → "large-v3-turbo")

### System Classification
- `ultra_precision`: High-accuracy processing
- `rapid_ultra`: Speed-optimized processing
- `segmented`: Large file chunked processing
- `large_file_ultra`: Enterprise large file handling

### Quality Types
- `reference`: Baseline quality results
- `high_quality`: Enhanced processing results
- `normal`: Standard processing results
- `draft`: Quick test results

### Test Size Categories
- `short`: <60 seconds
- `medium`: 60-600 seconds
- `long`: >600 seconds
- `unknown`: Duration not determinable

## 📈 Metadata Analysis

### Summary File Structure
```json
{
  "organization_date": "2025-06-02T00:56:22",
  "total_files_processed": 32,
  "statistics": {
    "models": {"large-v3": 3, "large-v3-turbo": 2, ...},
    "systems": {"ultra_precision": 6, "segmented": 3, ...},
    "test_sizes": {"short": 5, "medium": 7, ...},
    "quality_types": {"reference": 4, "normal": 21, ...}
  },
  "files": [
    {
      "file_name": "test_turbo_30s_speed_20250602.json",
      "model": "large-v3-turbo",
      "system": "benchmarks",
      "test_size": "short",
      "segments_count": 3,
      "average_confidence": 0.999999,
      "real_time_factor": 12.58,
      ...
    }
  ]
}
```

## 🎯 Use Cases

### 1. Performance Comparison
```bash
# Compare all turbo model tests
ls test_outputs/organized/models/large-v3-turbo/
```

### 2. System Benchmarking
```bash
# View all rapid processing results
ls test_outputs/organized/systems/rapid_ultra/
```

### 3. Quality Analysis
```bash
# Examine reference quality baselines
ls test_outputs/organized/reference/quality_baseline/
```

### 4. Test History
```bash
# Check metadata for test trends
jq '.statistics' test_outputs/organized/metadata_summary.json
```

## 🔍 Advanced Features

### File Association
- Automatically groups related files (.json, .csv, .srt)
- Maintains file relationships during organization
- Preserves original timestamps

### Metadata Extraction
- Parses JSON files for key metrics
- Calculates aggregate statistics
- Identifies test patterns and trends

### Error Handling
- Gracefully handles corrupted files
- Logs unprocessable files
- Maintains partial organization on errors

## 📝 Implementation Details

### Key Functions
```python
def categorize_file(file_path, file_data):
    """Categorize based on filename and content"""
    
def extract_file_metadata(file_path):
    """Extract metrics from JSON files"""
    
def organize_files(source_dir, target_dir):
    """Main organization logic"""
```

### Classification Priority
1. Filename patterns (most reliable)
2. JSON metadata fields
3. File characteristics
4. Default to uncategorized

## 🚀 Best Practices

### Regular Organization
```bash
# Add to cron for daily organization
0 2 * * * cd /project && python3 -m transcription.result_organizer
```

### Pre-commit Hook
```bash
# Organize before committing test results
python3 -m transcription.result_organizer && git add test_outputs/organized/
```

### Test Naming Convention
```
test_{model}_{duration}_{type}_{date}.{ext}
# Example: test_turbo_30s_speed_20250602.json
```

## 🔧 Troubleshooting

### Common Issues
1. **Permission errors**: Ensure write access to target directory
2. **Memory issues**: Process in batches for large datasets
3. **Corrupted files**: Check uncategorized folder

### Debug Mode
```bash
# Run with maximum verbosity
python3 -m transcription.result_organizer --verbose --dry-run
```

---

**Note**: Regular use of the result organizer ensures clean test management and enables effective performance tracking across model versions and system configurations.