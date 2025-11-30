from .security import SECURITY_PROMPT
from .performance import PERFORMANCE_PROMPT
from .observability import OBSERVABILITY_PROMPT
from .architecture import ARCHITECTURE_PROMPT

PROMPTS = {
    "security": SECURITY_PROMPT,
    "performance": PERFORMANCE_PROMPT,
    "observability": OBSERVABILITY_PROMPT,
    "architecture": ARCHITECTURE_PROMPT,
}

ALL_PROMPTS = [
    ("security", SECURITY_PROMPT),
    ("performance", PERFORMANCE_PROMPT),
    ("observability", OBSERVABILITY_PROMPT),
    ("architecture", ARCHITECTURE_PROMPT),
]
