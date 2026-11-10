"""The lab logbook: what this lab actually ran, and what happened.

Deliberately separate from `parallax.memory`. Memory holds the agent's own
conclusions and can be revised by the agent. The logbook holds primary records
written by chemists — they are evidence, not inference, and nothing but a human
should change them.

A logbook entry outranks published precedent for decisions in this lab. A paper
reporting that a coupling works does not overrule three failures on the bench
with the substrate in front of you, and the literature never publishes those
failures at all.
"""

from .store import (
    CONDITION_KEYS,
    Experiment,
    LogbookStore,
    add,
    date_tokens,
    normalise_conditions,
    outcomes_for_transform,
    search,
    stats,
    store,
)

__all__ = [
    "CONDITION_KEYS",
    "Experiment",
    "LogbookStore",
    "add",
    "date_tokens",
    "normalise_conditions",
    "outcomes_for_transform",
    "search",
    "stats",
    "store",
]
