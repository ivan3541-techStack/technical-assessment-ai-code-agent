"""Configuration settings for the AI Code Auditor."""

from pathlib import Path
from typing import Set

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_HEALTH_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 700
DEFAULT_MODEL = "codellama:7b"
DEFAULT_WORKERS = 6

# Generation Parameters
DEFAULT_NUM_PREDICT = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9

# File Processing
MAX_CODE_LENGTH = 6000
MIN_FILE_SIZE = 100

# Supported File Extensions
EXTENSIONS: Set[str] = {
    ".js", ".ts", ".tsx", ".jsx",
    ".py",
    ".java",
    ".go",
    ".rb",
    ".php",
    ".cs",
    ".kt",
    ".swift",
    ".rs"
}

# Directories to Exclude
EXCLUDE_DIRS: Set[str] = {
    ".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__",
    ".next", ".nuxt", ".cache", "coverage", "target", "out", "bin", "obj",
    ".mypy_cache", ".pytest_cache", "logs", "static", "public", ".idea", ".vscode",
}

# Configuration File Patterns
CONFIG_PATTERNS: Set[str] = {
    "package.json", "jest.config.js", "jest.config.ts", "vite.config.js", "vite.config.ts",
    "webpack.config.js", "tsconfig.json", ".eslintrc.js", ".prettierrc.js", "rollup.config.js",
    "babel.config.js", "next.config.js", ".env", ".env.local", ".gitignore", ".dockerignore",
}

# Project Metadata Files
PROJECT_META_CANDIDATES = [
    "package.json", "pyproject.toml", "requirements.txt", "go.mod",
    ".editorconfig", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
    "tsconfig.json", "sonar-project.properties",
    "CONTRIBUTING.md", "ARCHITECTURE.md",
    "jest.config.js", "vite.config.js", "webpack.config.js",
    "rollup.config.js", "babel.config.js", ".prettierrc.js", "next.config.js",
]

# Report Settings
REPORTS_DIR = Path("reports")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
REPORT_FILENAME_PATTERN = "technical_audit_report_{}.md"

