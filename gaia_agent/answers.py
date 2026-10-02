"""Normalise an answer for the scorer.

The leaderboard compares answers by EXACT MATCH and wants the bare answer only
(no "FINAL ANSWER" text), so cosmetic differences -- a trailing period, markdown
emphasis, wrapping quotes -- turn a right answer into a wrong one.
"""
import re

_WRAPPERS = ("**", "__", "`", '"', "'", "“", "”", "‘", "’")


def clean_answer(text: str) -> str:
    if not text:
        return text
    answer = text.strip()
    # Drop a leading "FINAL ANSWER:" if one slipped through.
    answer = re.sub(r"^\s*final answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    # Peel markdown emphasis / quotes that wrap the whole answer.
    changed = True
    while changed and len(answer) > 1:
        changed = False
        for w in _WRAPPERS:
            if answer.startswith(w) and answer.endswith(w) and len(answer) > 2 * len(w):
                answer = answer[len(w):-len(w)].strip()
                changed = True
    # Unbalanced emphasis left behind when a "**FINAL ANSWER: x**" line is split
    # on the marker. Asterisks/backticks are never part of an expected answer.
    answer = re.sub(r"^[*`]+|[*`]+$", "", answer).strip()
    # A sentence-ending period/exclamation is never part of the expected
    # answer, but keep an abbreviation's period ("U.S.", "St.") and decimals.
    if re.search(r"[A-Za-z0-9À-ɏ][.!]$", answer) and not re.search(r"(?:\b[A-Za-z]\.){2,}$", answer):
        answer = answer[:-1].rstrip()
    return answer
