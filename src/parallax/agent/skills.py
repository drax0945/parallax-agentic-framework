"""Skill loading for backends that do not load skills natively.

Claude Code and the Claude Agent SDK read `.claude/skills/*/SKILL.md` for
themselves. Every other model runtime needs to be told, so this module does the
same job explicitly: it advertises the skills up front by name and description,
and hands over a skill's full text only when the model asks for it.

That progressive disclosure matters — eleven skills is roughly 1,100 lines, far
too much to paste into every request, and most of it is irrelevant to any given
question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import settings

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path

    def summary_line(self) -> str:
        return f"- **{self.name}** — {self.description}"


def _parse(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = FRONTMATTER_RE.match(text)
    name, description = path.parent.name, ""
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip().strip('"').strip("'")
            if key == "name" and value:
                name = value
            elif key == "description" and value:
                description = value
        body = text[match.end() :]
    else:
        body = text
    return Skill(name=name, description=description, body=body.strip(), path=path)


def available_skills() -> list[Skill]:
    directory = settings().skills_dir
    if not directory.is_dir():
        return []
    skills = [s for s in (_parse(p) for p in sorted(directory.glob("*/SKILL.md"))) if s]
    return skills


def skill_index() -> str:
    """The up-front advertisement: names and descriptions only."""
    skills = available_skills()
    if not skills:
        return ""
    lines = [
        "# Available skills",
        "",
        "Each is a worked procedure for a kind of task. When one matches what you are "
        "doing, call `load_skill` to read it **before** you start, and follow it. They "
        "encode how to avoid the mistakes that matter in this domain.",
        "",
    ]
    lines += [s.summary_line() for s in skills]
    return "\n".join(lines)


def load_skill(name: str) -> dict[str, Any]:
    """Return one skill's full text."""
    skills = available_skills()
    exact = next((s for s in skills if s.name.lower() == name.lower()), None)
    if exact is None:
        partial = [s for s in skills if name.lower() in s.name.lower()]
        exact = partial[0] if len(partial) == 1 else None
    if exact is None:
        return {
            "error": f"no skill named {name!r}",
            "available": [s.name for s in skills],
        }
    return {
        "skill": exact.name,
        "description": exact.description,
        "instructions": exact.body,
        "note": "Follow this procedure for the current task.",
    }
