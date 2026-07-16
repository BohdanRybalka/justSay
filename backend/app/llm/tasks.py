"""Task-keyed LLM generation-config profiles.

Sibling to ``app/llm/config.py`` (which model/vendor/host to use) — this
module is the other, independent axis: how to sample, keyed by task
purpose. Cloud and Local providers resolve the same profile for a given
task and apply it uniformly; see docs/adr/010-llm-task-generation-profiles.md
for the full rationale.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProfile:
    temperature: float
    top_p: float
    max_tokens: int


DEFAULT_TASK = "dictation_cleanup"

TASK_PROFILES: dict[str, TaskProfile] = {
    "dictation_cleanup": TaskProfile(temperature=0.1, top_p=0.9, max_tokens=1024),
    "ai_prompt_structuring": TaskProfile(temperature=0.3, top_p=0.9, max_tokens=2048),
    "insights": TaskProfile(temperature=0.4, top_p=0.95, max_tokens=512),
}


def get_task_profile(task: str) -> TaskProfile:
    """Resolve a task name to its generation profile.

    Fail-soft: any name not in the table falls back to the default task's
    profile rather than raising, since callers pass loose strings (HTTP
    request bodies, `style` mappings) that should never crash a request.
    """
    return TASK_PROFILES.get(task, TASK_PROFILES[DEFAULT_TASK])
