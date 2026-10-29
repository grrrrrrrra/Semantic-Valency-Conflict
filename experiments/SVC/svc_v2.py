"""SVC v2 — specificity gate + predicate-conditioned valency decision.

Drop-in replacements for the NP-driven `detect_conflict_or_variability`
path in runner_ambik.py / runner_introplan.py. Pure functions, no LLM and
no spaCy imports, so this file is unit-testable on its own (see the
__main__ block at the bottom).

Three independent channels produce the help decision, and each one is
attributed in the raw log via `hr_source`:

  A  variant   — an underspecified actant with several admissible fillers
                 and nothing in the predicate to choose between them.
                 This is the preferences channel.
  B  hazard    — the predicate would act on a filler that is prohibited,
                 whether or not any choice was open. This is the safety
                 channel, and it must run on RESOLVED objects too: in
                 AmbiK, 81% of safety rows are not attribute-choice rows,
                 so an NP-driven pipeline cannot see them at all.
  C  adjunct   — an unresolved peripheral frame element with a physical
                 consequence. Narrowed, and logged separately so its
                 contribution can be ablated.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# normalisation

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]")
_ARTICLES = {"a", "an", "the", "some", "any"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = _PUNCT.sub(" ", text)
    tokens = [t for t in _WS.split(text) if t and t not in _ARTICLES]
    return " ".join(tokens)


def head_noun(obj: str) -> str:
    toks = normalize(obj).split()
    return toks[-1] if toks else ""


# channel A, part 1 — the specificity gate

def resolve_by_specificity(actant: str, matches: Iterable[str]):
    """Decide whether the actant's own modifiers already pick one match.

    Returns (resolved_object_or_None, remaining_candidates).

    An actant is resolved when its normalised string identifies exactly
    one candidate, either by equality or because the candidate contains
    it as a contiguous modifier-bearing span. A BARE head noun never
    resolves anything — "plate" is contained in every plate, which is the
    whole reason the choice is open.
    """
    matches = [m for m in matches if str(m).strip()]
    if not matches:
        return None, []

    a = normalize(actant)
    if not a:
        return None, matches

    exact = [m for m in matches if normalize(m) == a]
    if len(exact) == 1:
        return exact[0], []

    # a bare head noun carries no discriminating information
    if len(a.split()) < 2:
        return (matches[0], []) if len(matches) == 1 else (None, matches)

    # token subset, not substring: "ceramic plate" must narrow
    # "ceramic dinner plate", where the modifier is not contiguous with
    # the head. Substring matching silently misses every such case.
    a_tokens = set(a.split())
    contained = [m for m in matches if a_tokens <= set(normalize(m).split())]
    if len(contained) == 1:
        return contained[0], []

    if contained:
        # the modifiers narrowed the field without closing it —
        # carry the narrowed set forward, not the original one
        return None, contained

    return None, matches


# channel A, part 2 — the valency decision

REQUIRED = "required"
ADMISSIBLE = "admissible"
PROHIBITED = "prohibited"

_VALID_STATUSES = {REQUIRED, ADMISSIBLE, PROHIBITED}


@dataclass
class ActantDecision:
    actant: str
    predicate: str
    ask: bool
    channel: str | None
    reason: str
    candidates: list = field(default_factory=list)
    resolved: str | None = None


def _statuses(entries):
    """entries: parsed valency_prompt blocks. -> {status: [variant, ...]}"""
    out = {REQUIRED: [], ADMISSIBLE: [], PROHIBITED: []}
    for e in entries:
        if not isinstance(e, dict):
            continue
        variant = e.get("variant") or e.get("text")
        status = str(e.get("status", "")).strip().lower()
        if not variant:
            continue
        out.setdefault(status if status in _VALID_STATUSES else ADMISSIBLE, []).append(variant)
    return out


def decide_actant(actant, predicate, entries, resolved=None):
    """One actant's contribution to the help decision.

    `entries` are the valency_prompt blocks for this actant under this
    predicate. `resolved` is set when the specificity gate already closed
    the choice — the hazard branch still runs on it.
    """
    st = _statuses(entries)
    required, admissible, prohibited = st[REQUIRED], st[ADMISSIBLE], st[PROHIBITED]
    n = len(required) + len(admissible) + len(prohibited)
    safe = required + admissible

    if len(required) == 1:
        return ActantDecision(actant, predicate, False, None,
                              "predicate requires one specific filler; choice is determined",
                              safe, required[0])

    # channel B: hazard, independent of whether a choice was open
    if prohibited:
        if not safe:
            return ActantDecision(actant, predicate, True, "hazard",
                                  "every candidate filler is prohibited under this predicate",
                                  prohibited, resolved)
        return ActantDecision(actant, predicate, True, "hazard",
                              "a candidate filler is prohibited under this predicate",
                              prohibited + safe, resolved)

    # no open choice left
    if n <= 1:
        return ActantDecision(actant, predicate, False, None,
                              "actant resolves to a single admissible filler",
                              safe, resolved or (safe[0] if safe else None))

    if len(required) > 1:
        return ActantDecision(actant, predicate, True, "variant",
                              "predicate requires several mutually exclusive fillers",
                              required, resolved)

    if len(admissible) >= 2:
        return ActantDecision(actant, predicate, True, "variant",
                              "several fillers admissible, predicate does not choose between them",
                              admissible, resolved)

    return ActantDecision(actant, predicate, False, None,
                          "no open choice", safe, resolved)

# row-level aggregation

_CHANNEL_PRIORITY = ["hazard", "variant", "adjunct"]


def decide_row(actant_decisions, adjunct_hits=()):
    """-> (hr, hr_source, candidates, reason)

    hr_source is the highest-priority channel that fired, so the raw log
    carries the per-category breakdown of which stage is doing the work —
    that is the ablation table, produced for free by every run.
    """
    fired = {}
    candidates = []

    for d in actant_decisions:
        if d.ask:
            fired.setdefault(d.channel, d)
            candidates.extend(d.candidates)
        elif d.resolved:
            candidates.append(d.resolved)

    for hit in adjunct_hits:
        if hit.get("has_adjunct_ambiguity"):
            fired.setdefault("adjunct", hit)
            candidates.extend(hit.get("candidates") or [])

    for channel in _CHANNEL_PRIORITY:
        if channel in fired:
            src = fired[channel]
            reason = src.reason if isinstance(src, ActantDecision) else src.get("consequence", "")
            return 1, channel, _dedup(candidates), reason

    return 0, None, _dedup(candidates), "no channel fired"


def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        k = normalize(x)
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out


# self-test

if __name__ == "__main__":
    ok = True

    def check(label, got, want):
        global ok
        good = got == want
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {label}: {got!r}" + ("" if good else f" (want {want!r})"))

    plates = ["ceramic dinner plate", "plastic dinner plate", "glass salad plate",
              "ceramic salad plate"]
    check("bare head noun stays open",
          resolve_by_specificity("plate", plates)[0], None)
    check("full name resolves",
          resolve_by_specificity("the ceramic dinner plate", plates)[0], "ceramic dinner plate")
    check("partial modifier narrows but does not close",
          resolve_by_specificity("ceramic plate", plates)[1],
          ["ceramic dinner plate", "ceramic salad plate"])
    check("single match with bare noun resolves",
          resolve_by_specificity("whisk", ["whisk"])[0], "whisk")

    d = decide_actant("bowl", "microwave", [
        {"variant": "ceramic bowl", "frame": "Heatable_Container", "status": "admissible"},
        {"variant": "stainless steel bowl", "frame": "Metal_Container", "status": "prohibited"},
    ])
    check("safety fires on prohibited filler", (d.ask, d.channel), (True, "hazard"))

    d = decide_actant("stainless steel dinner plate", "heat in the microwave", [
        {"variant": "stainless steel dinner plate", "status": "prohibited"},
    ], resolved="stainless steel dinner plate")
    check("safety fires with no choice open", (d.ask, d.channel), (True, "hazard"))

    d = decide_actant("plate", "serve", [
        {"variant": "ceramic salad plate", "status": "admissible"},
        {"variant": "glass dinner plate", "status": "admissible"},
        {"variant": "plastic bread plate", "status": "admissible"},
    ])
    check("preferences fires on free choice", (d.ask, d.channel), (True, "variant"))

    d = decide_actant("appliance", "combine the yogurts", [
        {"variant": "blender", "status": "required"},
        {"variant": "whisk", "status": "admissible"},
    ])
    check("common sense suppressed by required filler", (d.ask, d.channel), (False, None))
    check("  and the required filler enters the candidate pool", d.resolved, "blender")

    d = decide_actant("knife", "cut the bread", [
        {"variant": "bread knife", "status": "required"},
        {"variant": "butter knife", "status": "prohibited"},
    ])
    check("required outranks a prohibited sibling", (d.ask, d.channel), (False, None))

    d = decide_actant("bowl", "microwave", [
        {"variant": "ceramic bowl", "status": "admissible"},
        {"variant": "steel bowl", "status": "prohibited"},
    ])
    check("hazard still fires without a required filler", (d.ask, d.channel), (True, "hazard"))

    resolved, rest = resolve_by_specificity("the paring knife",
                                            ["paring knife", "bread knife", "butter knife"])
    d = decide_actant("the paring knife", "cut", [], resolved=resolved)
    check("unambiguous suppressed by specificity gate", (d.ask, d.channel), (False, None))

    hr, src, cands, _ = decide_row([
        decide_actant("plate", "serve", [
            {"variant": "ceramic salad plate", "status": "admissible"},
            {"variant": "glass dinner plate", "status": "admissible"},
        ]),
        decide_actant("bowl", "microwave", [
            {"variant": "ceramic bowl", "status": "admissible"},
            {"variant": "stainless steel bowl", "status": "prohibited"},
        ]),
    ])
    check("hazard outranks variant in hr_source", (hr, src), (1, "hazard"))
    check("  candidate pool keeps both actants' fillers", len(cands), 4)

    hr, src, _, _ = decide_row([
        decide_actant("knife", "cut", [{"variant": "bread knife", "status": "required"},
                                       {"variant": "butter knife", "status": "admissible"}]),
    ], adjunct_hits=[{"has_adjunct_ambiguity": True, "consequence": "unwashed produce contaminates",
                      "candidates": ["wash first", "skip washing"]}])
    check("adjunct fires alone and is attributed", (hr, src), (1, "adjunct"))

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")