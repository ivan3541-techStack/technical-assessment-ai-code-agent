# Technical Assessment Code Auditor 🚀

Local AI-powered **code auditor** for technical assessment. Analyzes repositories using **Ollama + Llama 3.1:8B** with project-level context awareness for comprehensive code quality analysis.

## ✨ Features

- **Project Context Awareness**: Analyzes architecture, practices, and best practices alignment
- **3-Step Analysis Process**: 
  1. Project context extraction
  2. Best practices comparison
  3. File-level detailed audit
- **Parallel Processing**: 2-6 files simultaneously (configurable)
- **Llama 3.1:8B** (Recommended): Best balance of quality and performance
- **Rich Markdown Reports**: Structured issues with severity, risk, and recommendations
- **Centralized Configuration**: Easy customization via `config.py`
- **100% Local**: No cloud, fully private

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Adding Prompts](#adding-prompts)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## ⚡ Quick Start (5 minutes)

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Start Ollama server
ollama serve
```

### 2. Download Llama 3.1:8B (~4.7GB)
```bash
# Download recommended model
ollama pull llama3.1:8b

# Test it
ollama run llama3.1:8b "Explain what a code audit is"
```

**Alternative models tested:**
- ✅ **llama3.1:8b** - Recommended (best quality)
- ⚠️ `qwen2.5-coder:7b` - Good for code-specific tasks, but less accurate
- ⚠️ `codellama:7b` - Faster but less accurate

### 3. Configure Ollama (Recommended Settings)
```bash
# Add to ~/.zshrc or ~/.bashrc
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_CONTEXT_LENGTH=2048
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_GPU_OVERHEAD=512

# Apply settings
source ~/.zshrc  # or source ~/.bashrc
```

### 4. Setup Project
```bash
# Clone or download this repo
git clone <your-repo> technical-assessment-ai-code-agent
cd technical-assessment-ai-code-agent

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install requests
```

### 5. Run Your First Audit!
```bash
# Quick test (processes all files in repository)
python audit.py --path /path/to/your/repo

# With custom settings
python audit.py --path /path/to/your/repo \
  --model llama3.1:8b \
  --workers 4 \
  --out my-audit-report.md

# Verbose mode for debugging
python audit.py --path /path/to/your/repo --verbose
```


## 🏗️ Project Structure

```
technical-assessment-ai-code-agent/
├── audit.py              # Main orchestrator with parallel processing
├── config.py             # Centralized configuration constants
├── models.py             # Data models (Issue, Severity)
├── prompts/              # Modular AI prompts
│   ├── __init__.py       # Exports PROMPTS dict
│   ├── project_context.py
│   ├── best_practices.py
│   └── file_audit.py
├── reports/              # Generated Markdown reports
├── README.md
```

**Key Components:**

### 1. Configuration (`config.py`)
All settings in one place:
- Ollama endpoints and timeouts
- File extensions to analyze
- Directories to exclude
- Generation parameters
- Report settings

### 2. Main Auditor (`audit.py`)
Class-based architecture:
- `OllamaClient` - API communication
- `FileScanner` - Repository file discovery
- `ProjectAnalyzer` - Project context extraction
- `FileAuditor` - Individual file analysis
- `IssueAggregator` - Issue summarization
- `ReportGenerator` - Markdown report generation
- `AuditOrchestrator` - Overall coordination

### 3. Processing Flow
```
Repository → FileScanner → ProjectAnalyzer
    ↓
Project Context + Best Practices Analysis
    ↓
FileAuditor (ThreadPoolExecutor with N workers)
    ↓
Ollama API → JSON Parse → Issue Objects
    ↓
IssueAggregator → ReportGenerator → Markdown Report
```

**Supported Languages:** JavaScript/TypeScript, Python, Java, Go, Ruby, PHP, C#, Kotlin, Swift, Rust


## 🎯 Usage Examples

### Basic Usage
```bash
# Audit entire repository
python audit.py --path /path/to/your/repo

# With specific model
python audit.py --path /path/to/your/repo --model llama3.1:8b

# Custom output location
python audit.py --path /path/to/your/repo --out custom-report.md
```

### Performance Tuning
```bash
# More workers for faster processing (requires more RAM)
python audit.py --path /path/to/your/repo --workers 4

# Verbose mode for debugging
python audit.py --path /path/to/your/repo --verbose

# Complete example
python audit.py --path /path/to/your/repo \
  --model llama3.1:8b \
  --workers 2 \
  --out reports/my-audit.md \
  --verbose
```

### Configuration Customization
Edit `config.py` to customize:
```python
# Change file extensions to analyze
EXTENSIONS = {".js", ".ts", ".py", ".go"}

# Adjust code length limit
MAX_CODE_LENGTH = 10000

# Modify Ollama timeout
OLLAMA_TIMEOUT = 1000

# Change number of workers
DEFAULT_WORKERS = 4
```



## ⚙️ Ollama Configuration

### Recommended Settings (Tested Configuration)
Add these environment variables to your shell configuration:

```bash
# Add to ~/.zshrc or ~/.bashrc
export OLLAMA_NUM_PARALLEL=2           # 2 simultaneous requests
export OLLAMA_CONTEXT_LENGTH=2048      # Context window size
export OLLAMA_KEEP_ALIVE=30m           # Keep model in memory for 30 minutes
export OLLAMA_GPU_OVERHEAD=512         # 512MB system reserve

# Apply changes
source ~/.zshrc  # or source ~/.bashrc
```

### Performance Guidelines

| Workers | RAM Usage | Speed | Best For |
|---------|-----------|-------|----------|
| `--workers 2` | ~8GB | Standard | Most laptops (recommended) |
| `--workers 3` | ~10-11GB | Medium | 16GB+ RAM systems |
| `--workers 4` | ~12-13GB | Fast | 32GB+ RAM systems |
| `--workers 6+` | ~16GB+ | Very Fast | High-end workstations |

**Note:** Start with `--workers 2` and increase if you have sufficient RAM.

### Model Comparison (Tested)

| Model | Quality | Speed | RAM | Recommendation |
|-------|---------|-------|-----|----------------|
| **llama3.1:8b** | ⭐⭐⭐⭐⭐ | Medium | ~4.7GB | ✅ **Recommended** |
| qwen2.5-coder:7b | ⭐⭐⭐⭐ | Fast | ~4.1GB | Good for code-specific |
| codellama:7b | ⭐⭐⭐ | Very Fast | ~3.8GB | Faster but less accurate |

**Recommendation:** Use `llama3.1:8b` for best audit quality.

## ⚡ Performance Tuning

| Workers | RAM Usage | Speed | Recommendation |
|---------|-----------|-------|----------------|
| `--workers 2` | ~8GB | Standard | **Recommended for most laptops** |
| `--workers 3` | ~10-11GB | Medium | 16GB+ RAM systems |
| `--workers 4` | ~12-13GB | Fast | 32GB+ RAM systems |
| `--workers 6+` | ~16GB+ | Very Fast | High-end workstations |

**Tips:**
- Start with `--workers 2` for stability
- Monitor RAM usage with Activity Monitor / Task Manager
- Use `--verbose` flag to see detailed progress
- Larger codebases benefit from more workers


## 📊 Sample Report Output

Reports are generated in `reports/` directory with the following structure:

```markdown
# Technical Audit Report

**Repository**: `/path/to/your/repo`
**Model**: `llama3.1:8b`
**Generated**: 2024-12-14 16:24:35

## TL;DR
- Detected 12 high, 24 medium, 18 low, 5 informational issues.
- Most affected domains: Security (15), Performance (12), Architecture (8).

## Summary by domain
- **Security**: 15 issues (12 high, 3 medium). Focus on addressing high/medium items first.
- **Performance**: 12 issues (0 high, 9 medium). Focus on addressing high/medium items first.
- **Architecture**: 8 issues (0 high, 12 medium). Focus on addressing high/medium items first.

## Industry best practices vs project
| Domain | Project practice | Industry best practice | Gap | Severity |
|--------|------------------|------------------------|-----|----------|
| Security | Basic input validation | Comprehensive input sanitization + CSRF protection | Missing CSRF tokens, weak validation | high |

## Detailed issues
| File | Type | Severity | Risk | Description | Recommendation |
|------|------|----------|------|-------------|----------------|
| src/auth.js | Security | high | SQL injection vulnerability | User input directly in SQL query | Use parameterized queries |
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama not running` | Run `ollama serve` in a new terminal |
| `Model not found` | Run `ollama pull llama3.1:8b` |
| `Slow analysis` | Use `--workers 2` or reduce workers |
| `Out of RAM` | Close other apps or use `--workers 2` |
| `Connection timeout` | Increase `OLLAMA_TIMEOUT` in `config.py` |

**Remove old models:**
```bash
ollama list
ollama rm codellama:7b  # if you want to free up space
```

## 📄 License

MIT License. Uses **Llama 3.1:8B** under [Meta's Llama License](https://ollama.com/library/llama3.1).

**Commercial use OK** (no restrictions for code auditing purposes).

---

**Built for technical assessment at scale. 100% local, 100% private.** 🚀
