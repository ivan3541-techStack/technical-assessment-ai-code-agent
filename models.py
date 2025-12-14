# models.py
from dataclasses import dataclass
from enum import Enum
from typing import Literal, List, Optional

class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"

@dataclass
class Issue:
    file_path: str
    issue_type: str
    severity: Severity
    risk: str
    description: str
    recommendation: str
    evidence: Optional[str] = None
