#!/usr/bin/env python3
"""
Local AI Code Auditor with project-level context and structured issues.
Refactored version with improved architecture, separation of concerns, and maintainability.

Author: AI Code Auditor Team
Version: 2.0.0
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from config import (
    CONFIG_PATTERNS,
    DEFAULT_NUM_PREDICT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_WORKERS,
    EXCLUDE_DIRS,
    EXTENSIONS,
    MAX_CODE_LENGTH,
    MIN_FILE_SIZE,
    OLLAMA_HEALTH_URL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    PROJECT_META_CANDIDATES,
    REPORTS_DIR,
    REPORT_FILENAME_PATTERN,
    TIMESTAMP_FORMAT,
)
from models import Issue, Severity
from prompts import PROMPTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)



# ==== Utility Classes ====

class JSONExtractor:
    """Utility class for extracting JSON from LLM responses."""

    @staticmethod
    def extract_object(text: str) -> Optional[dict]:
        """
        Extract first JSON object {...} from text.

        Args:
            text: Text containing potential JSON object

        Returns:
            Parsed dictionary or None if extraction fails
        """
        start = text.find("{")
        if start == -1:
            return None

        brace_level = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                brace_level += 1
            elif ch == "}":
                brace_level -= 1
                if brace_level == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def extract_array(text: str) -> Optional[list]:
        """
        Extract JSON array from text, ignoring markdown code blocks.

        Args:
            text: Text containing potential JSON array

        Returns:
            Parsed list or None if extraction fails
        """
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        # Find first [ and last ]
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end <= start:
            return None

        snippet = text[start:end].strip()
        try:
            data = json.loads(snippet)
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, model: str):
        """
        Initialize Ollama client.

        Args:
            model: Name of the Ollama model to use
        """
        self.model = model

    @staticmethod
    def check_availability() -> bool:
        """
        Check if Ollama service is running.

        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            requests.get(OLLAMA_HEALTH_URL, timeout=5)
            return True
        except Exception:
            return False

    def generate(self, prompt: str, num_predict: Optional[int] = None) -> str:
        """
        Generate response from Ollama model.

        Args:
            prompt: Input prompt for the model
            num_predict: Maximum number of tokens to generate

        Returns:
            Model response text

        Raises:
            requests.HTTPError: If API request fails
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "num_predict": num_predict or DEFAULT_NUM_PREDICT,
            }
        }
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()


# ==== File and Project Analysis ====

class FileScanner:
    """Scans repository for code files to audit."""

    def __init__(self, repo_path: Path):
        """
        Initialize file scanner.

        Args:
            repo_path: Root path of repository to scan
        """
        self.repo_path = repo_path

    def scan(self, limit: Optional[int] = None) -> List[str]:
        """
        Scan repository for code files.

        Args:
            limit: Maximum number of files to process (None for no limit)

        Returns:
            List of file paths
        """
        files = []

        for root, dirs, files_in_dir in os.walk(self.repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files_in_dir:
                file_path = Path(root) / file

                # Skip config files
                if file_path.name in CONFIG_PATTERNS:
                    continue

                # Check extension and file size
                if (file_path.suffix.lower() in EXTENSIONS and
                    file_path.stat().st_size > MIN_FILE_SIZE):
                    files.append(str(file_path))

                    if limit and len(files) >= limit:
                        return files

        return files


class ProjectAnalyzer:
    """Analyzes project structure and context."""

    def __init__(self, repo_root: Path, ollama_client: OllamaClient):
        """
        Initialize project analyzer.

        Args:
            repo_root: Root path of repository
            ollama_client: Ollama client instance
        """
        self.repo_root = repo_root
        self.ollama_client = ollama_client
        self.json_extractor = JSONExtractor()

    def collect_metadata(self) -> Dict:
        """
        Collect metadata from project configuration files.

        Returns:
            Dictionary with metadata files and their content
        """
        files = []
        for rel in PROJECT_META_CANDIDATES:
            p = self.repo_root / rel
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    files.append({"path": rel, "content": content})
                except Exception as e:
                    logger.warning(f"Failed to read {rel}: {e}")

        return {"files": files}

    def build_context_prompt(self) -> str:
        """
        Build prompt for project context analysis.

        Returns:
            Formatted prompt string
        """
        meta = self.collect_metadata()
        files_block = "\n\n".join(
            f"### {f['path']}\n{f['content']}" for f in meta["files"]
        )
        return PROMPTS["project_context"] + "\n\n" + files_block

    def get_context(self) -> Dict:
        """
        Analyze project and extract context.

        Returns:
            Dictionary with architecture summary and practices
        """
        prompt = self.build_context_prompt()
        raw = self.ollama_client.generate(prompt)

        logger.debug("=== RAW PROJECT CONTEXT RESPONSE ===")
        logger.debug(raw)
        logger.debug("====================================")

        data = self.json_extractor.extract_object(raw)
        if isinstance(data, dict):
            return {
                "architecture_summary": data.get("architecture_summary", ""),
                "project_practices": data.get("project_practices", []),
                "raw": raw,
            }

        return {"architecture_summary": "", "project_practices": [], "raw": raw}

    def get_practices_diff(self, project_context: Dict) -> List[Dict]:
        """
        Compare project practices against industry best practices.

        Args:
            project_context: Project context dictionary

        Returns:
            List of practice differences
        """
        payload = {
            "architecture_summary": project_context.get("architecture_summary", ""),
            "project_practices": project_context.get("project_practices", []),
        }
        prompt = PROMPTS["best_practices"] + "\n\n" + json.dumps(
            payload, ensure_ascii=False, indent=2
        )
        raw = self.ollama_client.generate(prompt, num_predict=768)

        logger.debug("=== RAW BEST_PRACTICES RESPONSE ===")
        logger.debug(raw)
        logger.debug("===================================")

        data = self.json_extractor.extract_array(raw)
        return data if isinstance(data, list) else []


class FileAuditor:
    """Audits individual code files for issues."""

    def __init__(self, ollama_client: OllamaClient, repo_root: Path):
        """
        Initialize file auditor.

        Args:
            ollama_client: Ollama client instance
            repo_root: Root path of repository
        """
        self.ollama_client = ollama_client
        self.repo_root = repo_root
        self.json_extractor = JSONExtractor()

    def build_audit_prompt(self, project_context: Dict, file_path: str, code: str) -> str:
        """
        Build prompt for file audit.

        Args:
            project_context: Project context dictionary
            file_path: Relative path to file
            code: File content

        Returns:
            Formatted prompt string
        """
        if len(code) > MAX_CODE_LENGTH:
            code = code[:MAX_CODE_LENGTH] + "\n\n[... truncated ...]"

        ctx = {
            "architecture_summary": project_context.get("architecture_summary", ""),
            "project_practices": project_context.get("project_practices", []),
            "file_path": file_path,
            "code": code,
        }
        return PROMPTS["file_audit"] + "\n\n" + json.dumps(
            ctx, ensure_ascii=False, indent=2
        )

    def parse_issues(self, raw: str) -> List[Issue]:
        """
        Parse issues from model response.

        Args:
            raw: Raw model response

        Returns:
            List of Issue objects
        """
        data = self.json_extractor.extract_array(raw)
        if not isinstance(data, list):
            logger.debug(f"Invalid JSON array: {raw[:200]}...")
            return []

        issues: List[Issue] = []
        for i, item in enumerate(data):
            # Skip non-dict items (mistral bug)
            if not isinstance(item, dict):
                logger.warning(f"Issue {i}: expected dict, got {type(item)}")
                continue

            try:
                # Validate required fields
                required = ["file_path", "issue_type", "severity", "risk", "description"]
                if not all(k in item for k in required):
                    logger.warning(f"Issue {i} missing fields: {set(required) - set(item.keys())}")
                    continue

                # Validate severity enum
                if item["severity"] not in ["high", "medium", "low", "informational"]:
                    logger.warning(f"Issue {i} invalid severity: {item['severity']}")
                    continue

                issues.append(
                    Issue(
                        file_path=item["file_path"],
                        issue_type=item["issue_type"],
                        severity=Severity(item["severity"]),
                        risk=item["risk"],
                        description=item["description"],
                        recommendation=item.get("recommendation", ""),
                        evidence=item.get("evidence", ""),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Issue {i} parse error: {e}")
                continue

        return issues


    def audit_file(
        self,
        file_path: str,
        project_context: Dict
    ) -> Tuple[List[Issue], Optional[str]]:
        """
        Audit a single file.

        Args:
            file_path: Path to file
            project_context: Project context dictionary

        Returns:
            Tuple of (issues list, error message)
        """
        file_path_obj = Path(file_path)
        rel_path = os.path.relpath(file_path_obj, self.repo_root)

        try:
            code = file_path_obj.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return [], f"Cannot read file {rel_path}: {e}"

        prompt = self.build_audit_prompt(project_context, rel_path, code)
        try:
            raw = self.ollama_client.generate(prompt)
        except Exception as e:
            return [], f"Ollama error for {rel_path}: {e}"

        issues = self.parse_issues(raw)
        return issues, None


# ==== Report Generation ====

class IssueAggregator:
    """Aggregates and summarizes audit issues."""

    @staticmethod
    def build_tldr(issues: List[Issue]) -> List[str]:
        """
        Build TL;DR summary of issues.

        Args:
            issues: List of issues

        Returns:
            List of summary lines
        """
        if not issues:
            return [
                "No issues detected by the automated analyzer. "
                "Manual review still recommended."
            ]

        sev_counts = Counter(i.severity.value for i in issues)
        top_types = Counter(i.issue_type for i in issues).most_common(3)

        lines = [
            f"Detected {sev_counts.get('high', 0)} high, "
            f"{sev_counts.get('medium', 0)} medium, "
            f"{sev_counts.get('low', 0)} low, "
            f"{sev_counts.get('informational', 0)} informational issues."
        ]
        if top_types:
            types_str = ", ".join(f"{t} ({c})" for t, c in top_types)
            lines.append(f"Most affected domains: {types_str}.")
        return lines

    @staticmethod
    def build_domain_summary(issues: List[Issue]) -> Dict[str, str]:
        """
        Build summary by issue domain/type.

        Args:
            issues: List of issues

        Returns:
            Dictionary mapping domain to summary text
        """
        by_type = defaultdict(list)
        for i in issues:
            by_type[i.issue_type].append(i)

        summaries: Dict[str, str] = {}
        for t, lst in by_type.items():
            hs = sum(1 for i in lst if i.severity == Severity.HIGH)
            ms = sum(1 for i in lst if i.severity == Severity.MEDIUM)
            summaries[t] = (
                f"{len(lst)} issues ({hs} high, {ms} medium). "
                f"Focus on addressing high/medium items first."
            )
        return summaries


class ReportGenerator:
    """Generates markdown audit reports."""

    def __init__(self):
        """Initialize report generator."""
        self.aggregator = IssueAggregator()

    def generate(
        self,
        repo_path: str,
        model: str,
        project_context: Dict,
        practices_diff: List[Dict],
        all_issues: List[Issue],
        output_path: str,
    ) -> None:
        """
        Generate markdown audit report.

        Args:
            repo_path: Repository path
            model: Model name used
            project_context: Project context dictionary
            practices_diff: List of practice differences
            all_issues: All detected issues
            output_path: Output file path
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tldr = self.aggregator.build_tldr(all_issues)
        domain_summary = self.aggregator.build_domain_summary(all_issues)

        with open(output_path, "w", encoding="utf-8") as f:
            self._write_header(f, repo_path, model, ts)
            self._write_tldr(f, tldr)
            self._write_domain_summary(f, domain_summary)
            self._write_practices_diff(f, practices_diff)
            self._write_detailed_issues(f, all_issues)

    @staticmethod
    def _write_header(f, repo_path: str, model: str, timestamp: str) -> None:
        """Write report header."""
        f.write("# Technical Audit Report\n\n")
        f.write(f"**Repository**: `{repo_path}`  \n")
        f.write(f"**Model**: `{model}`  \n")
        f.write(f"**Generated**: {timestamp}\n\n")

    @staticmethod
    def _write_tldr(f, tldr: List[str]) -> None:
        """Write TL;DR section."""
        f.write("## TL;DR\n\n")
        for line in tldr:
            f.write(f"- {line}\n")
        f.write("\n")

    @staticmethod
    def _write_domain_summary(f, domain_summary: Dict[str, str]) -> None:
        """Write summary by domain section."""
        f.write("## Summary by domain\n\n")
        for domain, text in domain_summary.items():
            f.write(f"- **{domain}**: {text}\n")
        f.write("\n")

    @staticmethod
    def _sanitize_cell(text: str) -> str:
        """Sanitize text for markdown table cells."""
        return text.replace("|", "/").replace("\n", " ")

    def _write_practices_diff(self, f, practices_diff: List[Dict]) -> None:
        """Write best practices comparison section."""
        if not practices_diff:
            return

        f.write("## Industry best practices vs project\n\n")
        f.write("| Domain | Project practice | Industry best practice | Gap | Severity |\n")
        f.write("|--------|------------------|------------------------|-----|----------|\n")
        for row in practices_diff:
            f.write(
                f"| {row.get('domain', '')} "
                f"| {self._sanitize_cell(row.get('project_practice', ''))} "
                f"| {self._sanitize_cell(row.get('industry_best_practice', ''))} "
                f"| {self._sanitize_cell(row.get('gap', ''))} "
                f"| {row.get('severity', '')} |\n"
            )
        f.write("\n")

    def _write_detailed_issues(self, f, all_issues: List[Issue]) -> None:
        """Write detailed issues table."""
        f.write("## Detailed issues\n\n")
        f.write("| File | Type | Severity | Risk | Description | Recommendation |\n")
        f.write("|------|------|----------|------|-------------|----------------|\n")
        for issue in all_issues:
            f.write(
                f"| {issue.file_path} "
                f"| {issue.issue_type} "
                f"| {issue.severity.value} "
                f"| {self._sanitize_cell(issue.risk)} "
                f"| {self._sanitize_cell(issue.description)} "
                f"| {self._sanitize_cell(issue.recommendation)} |\n"
            )
        f.write("\n")


# ==== Main Orchestrator ====

class AuditOrchestrator:
    """Orchestrates the complete audit process."""

    def __init__(
        self,
        repo_path: str,
        model: str,
        workers: int = DEFAULT_WORKERS,
        output_path: Optional[str] = None
    ):
        """
        Initialize audit orchestrator.

        Args:
            repo_path: Path to repository
            model: Ollama model name
            workers: Number of parallel workers
            output_path: Optional output path for report
        """
        self.repo_path = Path(repo_path).resolve()
        self.model = model
        self.workers = workers
        self.output_path = output_path

        # Initialize components
        self.ollama_client = OllamaClient(model)
        self.file_scanner = FileScanner(self.repo_path)
        self.project_analyzer = ProjectAnalyzer(self.repo_path, self.ollama_client)
        self.file_auditor = FileAuditor(self.ollama_client, self.repo_path)
        self.report_generator = ReportGenerator()

    def run(self) -> int:
        """
        Run the complete audit process.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        # Validate Ollama is running
        if not OllamaClient.check_availability():
            logger.error("❌ Ollama is not running. Start it: ollama serve")
            return 1

        if not self.repo_path.exists():
            logger.error(f"❌ Repository not found: {self.repo_path}")
            return 1

        logger.info(f"🚀 Starting audit of {self.repo_path}")
        logger.info(f"📊 Model: {self.model}")
        logger.info(f"⚡ Workers: {self.workers}")

        # Setup output path
        REPORTS_DIR.mkdir(exist_ok=True)
        if not self.output_path:
            timestamp_str = datetime.now().strftime(TIMESTAMP_FORMAT)
            self.output_path = str(
                REPORTS_DIR /
                REPORT_FILENAME_PATTERN.format(timestamp_str)
            )

        start_time = datetime.now()

        # Step 1: Analyze project context
        logger.info("📦 Building project context...")
        project_context = self.project_analyzer.get_context()

        # Step 2: Compare against best practices
        logger.info("📐 Calculating best practices diff...")
        practices_diff = self.project_analyzer.get_practices_diff(project_context)

        # Step 3: Scan and audit files
        code_files = self.file_scanner.scan()
        logger.info(f"📁 Found {len(code_files)} files to audit")

        all_issues, errors = self._audit_files(code_files, project_context)

        # Report results
        total_time = datetime.now() - start_time
        logger.info(f"\n⏱️  Total time: {total_time}")
        logger.info(f"📊 Total issues: {len(all_issues)}")

        # Generate report
        logger.info(f"\n✍️  Generating report: {self.output_path}")
        self.report_generator.generate(
            repo_path=str(self.repo_path),
            model=self.model,
            project_context=project_context,
            practices_diff=practices_diff,
            all_issues=all_issues,
            output_path=self.output_path,
        )

        logger.info(f"✅ Audit complete! Report: {self.output_path}")
        if errors:
            logger.warning(
                f"⚠️  There were {len(errors)} file errors. See console output above."
            )

        return 0

    def _audit_files(
        self,
        code_files: List[str],
        project_context: Dict
    ) -> Tuple[List[Issue], List[str]]:
        """
        Audit multiple files in parallel.

        Args:
            code_files: List of file paths to audit
            project_context: Project context dictionary

        Returns:
            Tuple of (all issues, error messages)
        """
        all_issues: List[Issue] = []
        errors: List[str] = []

        logger.info(f"📝 Processing {len(code_files)} file(s)...")

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self.file_auditor.audit_file,
                    file_path,
                    project_context
                ): file_path
                for file_path in code_files
            }

            completed = 0
            for future in as_completed(futures):
                issues, err = future.result()
                completed += 1

                if err:
                    errors.append(err)
                    logger.warning(f"⚠️  {err}")
                else:
                    all_issues.extend(issues)
                    logger.info(
                        f"✅ [{completed}/{len(code_files)}] issues found: {len(issues)}"
                    )

        return all_issues, errors


# ==== CLI Entry Point ====

def main() -> int:
    """
    Main entry point for the CLI application.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Local AI Code Auditor with project context analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --path /path/to/repo
  %(prog)s --path /path/to/repo --model codellama:13b --workers 8
  %(prog)s --path /path/to/repo --out custom_report.md
        """
    )
    parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to repository for audit"
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="Path to output Markdown report"
    )
    parser.add_argument(
        "--model",
        default="codellama:7b",
        help="Ollama model name (default: codellama:7b)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Max parallel tasks (default: {DEFAULT_WORKERS})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create and run orchestrator
    orchestrator = AuditOrchestrator(
        repo_path=args.path,
        model=args.model,
        workers=args.workers,
        output_path=args.out
    )

    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())

