# Technical Assessment Code Auditor 🚀

Local AI-powered **code auditor** for technical assessment. Analyzes repositories using **Ollama + CodeLlama:7B** for security, performance, observability, and architecture issues.

## ✨ Features

- **4 Audit Categories(Adjustable)**: Security, Performance, Observability, **Architecture**
- **Parallel Processing**: 2-8 files simultaneously
- **CodeLlama:7B**: **3-4x faster** than DeepSeek-Coder-V2 (~4GB RAM)
- **Rich Markdown Reports**: Issues + fixes + severity ratings
- **Configurable**: Select prompts, models, workers
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
macOS / Linux
`curl -fsSL https://ollama.com/install.sh | sh`

Verify installation
`ollama --version`
Start Ollama server
`ollama serve`


### 2. Install CodeLlama:7B (~4GB)
Download model
`ollama pull codellama:7b`

Test it
`ollama run codellama:7b "Write hello world in Python"`


### 3. Setup Project
Clone or download this repo
`git clone <your-repo> technical-assessment-ai-code-agent`

Go to project folder
`cd technical-assessment-ai-code-agent`

Python environment

`python3 -m venv venv`

`source venv/bin/activate` # macOS/Linux

`pip3 install requests`


### 4. Run Your First Audit!
Test architecture (5-10 min)
# Test architecture only (~15 min for 234 files)
python3 audit.py --path ~/audits/repo-to-audit \
  --prompts architecture \
  --model codellama:7b \
  --workers 3

# Full audit (~90 min for 234 files × 4 prompts)
python3 audit.py --path ~/audits/gaf-rps-services \
  --model codellama:7b \
  --workers 3


## 🏗️ Project Structure
```
code-audit-agent/
├── audit.py # Main CLI + parallel processing
├── prompts/ # Modular AI prompts
│ ├── init.py # Exports PROMPTS dict
│ ├── security.py
│ ├── performance.py
│ ├── observability.py
│ └── architecture.py
├── reports/ # Generated Markdown reports
└── README.md
```


**Processing Flow:**
```
Repository → iter_code_files() → ThreadPoolExecutor (3 workers)
    ↓
audit_task() → Ollama API → JSON Parse
    ↓
generate_report() → Markdown Report
```


## 🎯 Usage Examples

Full audit (all prompts)
`python3 audit.py --path ~/audits/gaf-rps-services --workers 3`

Single category
`python3 audit.py --path ~/audits/gaf-rps-services --prompts security --workers 3`

Custom output + model
`python3 audit.py --path ~/audits/gaf-rps-services \
  --out custom-report.md \
  --model codellama:7b`

Multiple prompts
`python3 audit.py --path ~/audits/gaf-rps-services \
  --prompts architecture security performance`

Max speed (M1/M2 Mac)
`python3 audit.py --path /path/to/repo --workers 8`

## 🎯 Ollama Configuration
M1 Pro 16GB (Recommended)

```
export OLLAMA_NUM_PARALLEL=2           # 2 simultaneous requests
export OLLAMA_KEEP_ALIVE=150m          # Unload after 150 min
export OLLAMA_GPU_OVERHEAD=512         # 512MB system reserve
export OLLAMA_CONTEXT_LENGTH=2048      # Limit context window
```

M1 Max 64GB (Aggressive)

```
export OLLAMA_NUM_PARALLEL=6
export OLLAMA_KEEP_ALIVE=150m
export OLLAMA_GPU_OVERHEAD=512
export OLLAMA_CONTEXT_LENGTH=4096
```


**Supported Languages:** JS/TS/JSX, Python, Java, Go, Ruby, PHP, C#, Kotlin, Swift, Rust

## ➕ Adding New Prompts

### 1. Create `prompts/custom.py`
```
CUSTOM_PROMPT = """
You are a senior [ROLE] reviewing [FOCUS AREA].

Return JSON:
{
"issues": [{"title": "", "severity": "low|medium|high|critical", "fix": ""}]
}

Code: {{code}}
File: {{filename}}
"""
```

### 2. Update `prompts/__init__.py`
```
from .custom import CUSTOM_PROMPT

PROMPTS["custom"] = CUSTOM_PROMPT
ALL_PROMPTS.append(("custom", CUSTOM_PROMPT))
```

### 3. Use it
`python3 audit.py --path /path/to/repo --prompts custom`

## ⚡ Performance Tuning

| Workers | Speed | RAM | Recommendation |
|---------|-------|-----|----------------|
| `--workers 2` | **8GB** | Low | **Default** |
| `--workers 3` | **10-11GB** | Medium | M1/M2 Mac |
| `--workers 4` | **12-13GB** | High | M3 Max+ |


## 📊 Sample Report Output

**`reports/audit_20251130_1624.md`:**
Technical Audit Report
Repository: /Users/apple/audits/gaf-rps-services
Model: codellama:7b
Files analyzed: 256

src/services/userService.js
Architecture
1. SRP Violation (medium)
   Explanation: Handles auth + email + database logic
   Fix: Split into UserAuthService, UserEmailService

Architecture score: C

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ollama not running` | `ollama serve` (new terminal) |
| `Model not found` | `ollama pull codellama:7b` |
| `Slow analysis` | Use `--workers 2` or `--model phi3:mini` |
| `Out of RAM` | Close apps or use `--workers 2` |

**Remove old models:**
ollama list
ollama rm deepseek-coder-v2

## 📄 License

MIT License. Uses **CodeLlama:7B** under [Llama 2 Community License](https://ollama.com/library/codellama:7b).

**Commercial use OK** (no military/hacking). [web:90][web:93]

---

**Built for technical assessment at scale. 100% local, 100% private.** 🚀
