PERFORMANCE_PROMPT = """
You are a principal backend engineer reviewing performance aspects.
Analyze the following code for N+1 queries, blocking operations, timeouts.

Return JSON:
{
  "issues": [{"title": "", "impact": "low|medium|high", "explanation": "", "fix": ""}]
}

Code:
File: {{filename}}
"""
