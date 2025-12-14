BEST_PRACTICES_PROMPT = """
Compare the provided project practices to industry best practices.

Input:
- architecture_summary
- project_practices (array)

TASK:
For each project practice, create one comparison entry.

Return ONLY a JSON ARRAY in this format:

[
  {
    "domain": "one of: architecture|testing|observability|security|performance|maintainability",
    "project_practice": "exact string from input project_practices",
    "industry_best_practice": "industry standard description",
    "gap": "actual gap found in THIS project",
    "severity": "high|medium|low|informational"
  }
]

IMPORTANT:
- Start with [ and end with ]
- One object per project practice
- Analyze THIS specific project
"""
