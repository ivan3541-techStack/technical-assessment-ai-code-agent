#!/usr/bin/env python3
"""
Local AI Code Auditor with OPTIMIZED parallel processing.
Single-level ThreadPoolExecutor for better performance.
"""

import os
import json
import textwrap
import requests
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import ALL_PROMPTS, PROMPTS

OLLAMA_URL = "http://localhost:11434/api/generate"
EXTENSIONS = {".js", ".ts", ".tsx", ".jsx", ".py", ".java", ".go", ".rb", ".php", ".cs", ".kt", ".swift", ".rs"}

# ✅ Configurable exclusions
EXCLUDE_DIRS = {
    ".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__",
    ".next", ".nuxt", ".cache", "coverage", "target", "out", "bin", "obj",
    ".mypy_cache", ".pytest_cache", "logs", "static", "public", ".idea", ".vscode",
}

CONFIG_PATTERNS = {
    "package.json", "jest.config.js", "jest.config.ts", "vite.config.js", "vite.config.ts",
    "webpack.config.js", "tsconfig.json", ".eslintrc.js", ".prettierrc.js", "rollup.config.js",
    "babel.config.js", "next.config.js", ".env", ".env.local", ".gitignore", ".dockerignore",
}


def iter_code_files(repo_path: str) -> List[str]:
    """Iterate through code files (skip config files and excluded dirs)."""
    files = []
    repo = Path(repo_path)

    for root, dirs, files_in_dir in os.walk(repo):
        # ✅ Exclude directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files_in_dir:
            file_path = Path(root) / file

            # Skip config files
            if file_path.name in CONFIG_PATTERNS:
                continue

            # Only code files with minimum size
            if file_path.suffix.lower() in EXTENSIONS and file_path.stat().st_size > 100:
                files.append(str(file_path))

    return files


def call_ollama(prompt: str, model: str) -> str:
    """Request to local Ollama."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=700)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"ERROR: {e}"


def build_prompt(template: str, code: str, filename: str, context: str = "") -> str:
    """Build prompt with variable substitution."""
    max_code_len = 6000
    if len(code) > max_code_len:
        code = code[:max_code_len] + "\n\n[... truncated ...]"

    prompt = template.replace("{{code}}", code)
    prompt = prompt.replace("{{filename}}", filename)
    prompt = prompt.replace("{{context}}", context)
    return prompt


def audit_task(task: Tuple[str, str, str, str, str]) -> Tuple[str, str, Dict]:
    """
    Single audit task: (file_path, repo_root, prompt_name, prompt_template, model).
    FLAT parallelization: no nested ThreadPoolExecutor.
    """
    file_path, repo_root, prompt_name, prompt_template, model = task

    rel_path = os.path.relpath(file_path, repo_root)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception as e:
        return rel_path, prompt_name, {"error": f"Cannot read file: {e}"}

    try:
        prompt = build_prompt(prompt_template, code, rel_path)
        raw_response = call_ollama(prompt, model)

        # Parse JSON
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = raw_response[start:end]
                parsed = json.loads(json_str)
            else:
                parsed = {"raw": raw_response}
        except json.JSONDecodeError:
            parsed = {"raw": raw_response}

        return rel_path, prompt_name, parsed
    except Exception as e:
        return rel_path, prompt_name, {"error": str(e)}


def generate_report(all_results: List[Tuple[str, Dict]], repo_path: str, output_path: str,
                    model: str, selected_prompts: List[Tuple[str, str]]):
    """Generate Markdown report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Technical Audit Report\n\n")
        f.write(f"**Repository**: `{repo_path}`\n")
        f.write(f"**Model**: `{model}`\n")
        f.write(f"**Generated**: {ts}\n")
        f.write(f"**Files analyzed**: {len(all_results)}\n")
        f.write(f"**Prompts used**: {len(selected_prompts)}\n\n")

        f.write("## Summary\n\n")
        total_files = len(all_results)
        f.write(f"- **Total files**: {total_files}\n")
        f.write(f"- **Prompts**: {[name for name, _ in selected_prompts]}\n\n")

        for rel_path, results in all_results:
            f.write(f"## {rel_path}\n\n")

            for section_name, data in results.items():
                if "error" in data:
                    f.write(f"### {section_name.title()}\n")
                    f.write(f"_Error: {data['error']}_\n\n")
                    continue

                f.write(f"### {section_name.title()}\n\n")

                if "issues" in data and data["issues"]:
                    for i, issue in enumerate(data["issues"], 1):
                        f.write(f"**{i}. {issue.get('title', 'Issue')}**\n\n")
                        sev = issue.get("severity") or issue.get("impact", "unknown")
                        f.write(f"*Severity*: **{sev}**\n\n")
                        if "explanation" in issue:
                            f.write(f"**Explanation**:\n{issue['explanation']}\n\n")
                        if any(k in issue for k in ["fix", "recommendation", "refactor"]):
                            fix_key = next((k for k in ["fix", "recommendation", "refactor"] if k in issue), None)
                            f.write(f"**Fix**:\n{issue[fix_key]}\n\n")

                elif "violations" in data and data["violations"]:
                    for violation in data["violations"]:
                        f.write(f"- **{violation.get('principle', '')}: {violation.get('title', 'Violation')}** "
                                f"({violation.get('severity', 'unknown')})\n")
                        f.write(f"  *Impact*: {violation.get('impact', '')}\n")
                        f.write(f"  *Fix*: {violation.get('refactor', '')}\n\n")

                elif "raw" in data:
                    f.write("```\n")
                    f.write(textwrap.fill(data["raw"], width=100, break_long_words=False))
                    f.write("\n```\n\n")
                else:
                    f.write("_No issues found._\n\n")

            f.write("\n---\n\n")


def main():
    parser = argparse.ArgumentParser(description="Local AI Code Auditor (Optimized)")
    parser.add_argument("--path", "-p", required=True, help="Path to repository for audit")
    parser.add_argument("--out", "-o", default=None, help="Path to Markdown report")
    parser.add_argument("--model", default="codellama:7b", help="Ollama model name")
    parser.add_argument("--prompts", nargs="*", default=None, help="List of prompts (security,performance,...)")
    parser.add_argument("--workers", type=int, default=6, help="Max parallel tasks (default: 6)")

    args = parser.parse_args()

    selected_model = args.model

    # Check Ollama
    try:
        requests.get("http://localhost:11434", timeout=5)
    except Exception:
        print("❌ Ollama is not running. Start it: ollama serve")
        return 1

    repo_path = os.path.abspath(args.path)
    if not os.path.exists(repo_path):
        print(f"❌ Repository not found: {repo_path}")
        return 1

    # Select prompts
    selected_prompts = ALL_PROMPTS
    if args.prompts:
        selected_prompts = [(name, PROMPTS[name]) for name in args.prompts if name in PROMPTS]
        if not selected_prompts:
            print("❌ Unknown prompts. Available:", ", ".join(PROMPTS.keys()))
            return 1

    print(f"🚀 Starting OPTIMIZED audit of {repo_path}")
    print(f"📊 Model: {selected_model}")
    print(f"🎯 Prompts: {[name for name, _ in selected_prompts]}")
    print(f"⚡ Workers: {args.workers}")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    out_path = args.out or reports_dir / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    start_time = datetime.now()

    code_files = iter_code_files(repo_path)
    print(f"📁 Found files: {len(code_files)}")

    # ✅ CREATE FLAT TASK LIST (no nested loops)
    tasks = []
    for file_path in code_files:
        for prompt_name, prompt_template in selected_prompts:
            tasks.append((file_path, repo_path, prompt_name, prompt_template, selected_model))

    print(f"📝 Total tasks: {len(tasks)} (files × prompts)")

    # ✅ SINGLE-LEVEL PARALLELIZATION
    all_results_raw = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_task, task): task for task in tasks}

        completed = 0
        for future in as_completed(futures):
            try:
                rel_path, prompt_name, result = future.result()
                all_results_raw.append((rel_path, prompt_name, result))
                completed += 1
                print(f"✅ [{completed}/{len(tasks)}] {rel_path} ({prompt_name})")
            except Exception as e:
                print(f"❌ Error: {e}")

    # ✅ MERGE RESULTS BY FILE
    all_results = {}
    for rel_path, prompt_name, result in all_results_raw:
        if rel_path not in all_results:
            all_results[rel_path] = {}
        all_results[rel_path][prompt_name] = result

    all_results = list(all_results.items())

    total_time = datetime.now() - start_time
    print(f"\n⏱️  Total time: {total_time}")

    print(f"\n✍️ Generating report: {out_path}")
    generate_report(all_results, repo_path, out_path, selected_model, selected_prompts)

    print(f"✅ Audit complete! Report: {out_path}")
    print(f"📊 Files analyzed: {len(all_results)}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
