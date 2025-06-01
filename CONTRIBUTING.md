# 🤝 Contributing to Ultra Audio Transcription

We welcome contributions to Ultra Audio Transcription! This document provides guidelines for contributing to the project.

## 🎯 Getting Started

### Prerequisites

- Python 3.8+
- Git
- GPU with CUDA support (recommended)
- 8GB+ RAM (16GB+ recommended)

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/ibushimaru/ultra-transcription.git
cd ultra-transcription

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## 📋 How to Contribute

### 1. Issues and Bug Reports

- **Check existing issues** first to avoid duplicates
- **Use issue templates** when available
- **Provide detailed information** including:
  - System configuration (OS, GPU, CUDA version)
  - Audio file details (format, duration, language)
  - Expected vs actual behavior
  - Complete error messages and stack traces

### 2. Feature Requests

- **Check roadmap** in README.md first
- **Describe the use case** clearly
- **Consider performance implications**
- **Provide examples** when possible

### 3. Pull Requests

#### Before You Start
- **Open an issue** to discuss major changes
- **Check existing PRs** to avoid duplicates
- **Read the code style guidelines** below

#### PR Process
1. **Create a feature branch** from `main`
2. **Write tests** for new functionality
3. **Ensure all tests pass**
4. **Update documentation** as needed
5. **Submit the PR** with a clear description

## 🔧 Development Guidelines

### Code Style

We use the following tools for code quality:

```bash
# Format code
black transcription/
isort transcription/

# Lint code
flake8 transcription/
mypy transcription/

# Run all checks
pre-commit run --all-files
```

#### Python Style Guidelines
- **PEP 8** compliance with 100-character line limit
- **Type hints** for all public functions
- **Comprehensive docstrings** using Google style
- **Descriptive variable names**
- **Clear error messages**

#### Example Function Documentation
```python
def process_audio(
    audio_file: str, 
    model: str = "large-v3",
    enable_gpu: bool = True
) -> TranscriptionResult:
    """
    Process audio file with GPU-accelerated transcription.
    
    Args:
        audio_file: Path to audio file (MP3, WAV, etc.)
        model: Whisper model size to use
        enable_gpu: Whether to use GPU acceleration
        
    Returns:
        TranscriptionResult with segments and metadata
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        CUDAError: If GPU acceleration fails
        
    Example:
        >>> result = process_audio("meeting.mp3", model="large-v3")
        >>> print(f"Accuracy: {result.average_confidence:.1%}")
    """
```

### Testing Guidelines

#### Test Structure
```bash
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for component interaction
├── performance/             # Performance and benchmark tests
└── fixtures/               # Test data and utilities
```

#### Writing Tests
- **Use pytest** for all tests
- **Include type hints** in test functions
- **Test both success and failure cases**
- **Mock external dependencies**
- **Use descriptive test names**

#### Test Example
```python
def test_gpu_ultra_precision_accuracy():
    """Test GPU Ultra Precision achieves expected accuracy."""
    audio_file = "tests/fixtures/test_audio_90s.mp3"
    
    result = process_gpu_ultra_precision(
        audio_file=audio_file,
        model_list=["large-v3"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    assert result.average_confidence >= 0.90
    assert len(result.segments) > 0
    assert all(seg.speaker_id != "SPEAKER_UNKNOWN" for seg in result.segments)
```

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=transcription --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m gpu            # GPU tests only (requires CUDA)

# Run performance tests
pytest tests/performance/ -v
```

### Performance Considerations

#### GPU Development
- **Always test on both CPU and GPU**
- **Optimize memory usage** - use `torch.cuda.empty_cache()`
- **Handle GPU unavailability gracefully**
- **Monitor VRAM usage** during development

#### Memory Management
- **Profile memory usage** for large files
- **Use streaming processing** for very long audio
- **Clean up temporary files**
- **Test with various file sizes**

#### Speaker Recognition
- **Test with multi-speaker audio**
- **Validate speaker consistency**
- **Check accuracy metrics**
- **Test different speaker counts**

## 📊 Component-Specific Guidelines

### 1. Audio Processing Components
- **Maintain audio quality** during preprocessing
- **Handle various audio formats**
- **Test with different sample rates**
- **Validate noise reduction effectiveness**

### 2. Transcription Engines
- **Ensure model compatibility**
- **Validate confidence scores**
- **Test language-specific features**
- **Benchmark processing speed**

### 3. Speaker Diarization
- **Test speaker consistency algorithms**
- **Validate multiple speaker methods**
- **Check temporal consistency**
- **Test edge cases (single speaker, many speakers)**

### 4. Output Formatters
- **Validate all output formats**
- **Test schema compliance**
- **Ensure data integrity**
- **Check backward compatibility**

## 🚀 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes to API
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Performance benchmarks run
- [ ] GPU compatibility tested

## 🐛 Debugging Guidelines

### Common Issues

#### CUDA/GPU Problems
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi

# Test with CPU fallback
ultra-transcribe audio.mp3 --device cpu
```

#### Memory Issues
```bash
# Monitor memory usage
htop

# Reduce GPU memory fraction
ultra-transcribe audio.mp3 --gpu-memory-fraction 0.6

# Use smaller models
ultra-transcribe audio.mp3 --model medium
```

#### Speaker Recognition Issues
```bash
# Try different methods
ultra-transcribe audio.mp3 --speaker-method acoustic
ultra-transcribe audio.mp3 --speaker-method clustering

# Disable speaker consistency temporarily
ultra-transcribe audio.mp3 --no-speaker-consistency
```

### Debug Information
When reporting bugs, include:

```bash
# System information
python --version
pip list | grep -E "(torch|whisper|librosa|pyannote)"

# GPU information (if applicable)
nvidia-smi

# Test command that reproduces the issue
ultra-transcribe test.mp3 --verbose
```

## 📖 Documentation

### Documentation Standards
- **Keep README.md updated** with new features
- **Update API documentation** for public methods
- **Include code examples** in docstrings
- **Write clear, concise explanations**

### Documentation Structure
```
docs/
├── ARCHITECTURE.md      # System design
├── API_REFERENCE.md     # Programming interface
├── USER_MANUAL.md       # Usage guide
├── TROUBLESHOOTING.md   # Common issues
└── DEVELOPER_GUIDE.md   # Contributing guide
```

## 🎯 Areas for Contribution

### High Priority
- **Performance optimizations** for specific GPU architectures
- **Memory usage improvements** for large files
- **Speaker recognition accuracy** enhancements
- **Multi-language support** expansion

### Medium Priority
- **New output formats** for specific use cases
- **Integration tools** for common workflows
- **Quality metrics** and analysis tools
- **Error handling** improvements

### Low Priority
- **UI/GUI development** for desktop application
- **Cloud deployment** tools and scripts
- **Mobile optimization** research
- **Advanced audio preprocessing** techniques

## 🤝 Community

### Communication
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

### Code of Conduct
We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## 🔍 Review Process

### What We Look For
- **Code quality**: Clean, readable, well-documented code
- **Test coverage**: Comprehensive tests for new functionality
- **Performance**: No significant performance regressions
- **Compatibility**: Works across supported platforms
- **Documentation**: Clear documentation for new features

### Review Timeline
- **Initial review**: Within 3-5 business days
- **Follow-up reviews**: Within 1-2 business days
- **Final approval**: After all requirements met

## 🙏 Recognition

Contributors will be recognized in:
- **CHANGELOG.md** for their contributions
- **README.md** acknowledgments section
- **GitHub releases** notes

## 📞 Getting Help

If you need help:
1. **Check the documentation** first
2. **Search existing issues** for similar problems
3. **Ask in GitHub Discussions** for general questions
4. **Open an issue** for bugs or specific problems

Thank you for contributing to Ultra Audio Transcription! 🚀