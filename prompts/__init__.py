from .project_context import PROJECT_CONTEXT_PROMPT
from .best_practices import BEST_PRACTICES_PROMPT
from .file_audit import FILE_AUDIT_PROMPT

PROMPTS = {
    "project_context": PROJECT_CONTEXT_PROMPT,
    "best_practices": BEST_PRACTICES_PROMPT,
    "file_audit": FILE_AUDIT_PROMPT,
}

ALL_PROMPTS = [
    ("project_context", PROJECT_CONTEXT_PROMPT),
    ("best_practices", BEST_PRACTICES_PROMPT),
    ("file_audit", FILE_AUDIT_PROMPT),
]
