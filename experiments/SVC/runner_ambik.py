import os
import spacy
import re
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from functools import lru_cache
import json
import nltk
import yaml

from llm import load_llm
import llm as _llm_module  
import parser as response_parser
import metrics

import nltk

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

import argparse
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svc_v2

nlp = spacy.load("en_core_web_lg")
embedding_model  = SentenceTransformer('paraphrase-mpnet-base-v2')

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--model", dest="svc_model",
                 default=os.environ.get("SVC_MODEL", "qwen3-32b"),
                 help="key from llm_config.yaml's models: registry")
_ap.add_argument("--limit", dest="svc_limit", type=int, default=None,
                 help="process only the first N rows (smoke test)")
_cli_args, _ = _ap.parse_known_args()

MODEL_NAME = _cli_args.svc_model
model = load_llm(MODEL_NAME)
print(f"[svc] backbone = {MODEL_NAME}")

def load_config(config_path=None):

    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

matching_config = config["matching"]
weight_map = config["weights"]
manual_synonyms = config["manual_synonyms"]

EXTRACT_STRINGS_INCLUDE_LLM_VARIANTS = True

USE_MANUAL_SYNONYMS_IN_LLM_PROMPT = False

SPECIFICITY_GATE = True

RUN_HAZARD_CHECK = False
HAZARD_CALLS_PER_ROW = 4
RUN_ADJUNCT_CHECK = False
ADJUNCT_CALLS_PER_ROW = 3

def preprocess_text(text):
    doc = nlp(text)
    return doc

def extract_tokens_info(doc):
    tokens_info = []
    for token in doc:
        tokens_info.append({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'dep': token.dep_,
            'head': token.head.text,
            'is_stop': token.is_stop
        })
    return tokens_info

def identify_predicates_from_doc(doc, debug = False):
    predicates = []
    seen = set()

    STRONG_DEPS = {
        'ROOT', 'dobj', 'nsubj', 'nsubjpass', 'attr',
        'acl', 'relcl', 'advcl', 'xcomp', 'ccomp', 'conj'
    }
    LIGHT_VERBS = {'make', 'take', 'have', 'give', 'do'}
    SPATIAL_PREPS = {'on', 'in', 'at', 'under', 'over', 'beside', 'left', 'right'}

    for token in doc:
        if token.pos_ == 'ADP' and token.lemma_ in SPATIAL_PREPS:
            if debug:
                print(f"Skip spatial preposition: {token.text}")
            continue

        if token.pos_ not in {'VERB', 'NOUN', 'ADJ'}:
            if debug:
                print(f"Skip POS: {token.text} ({token.pos_})")
            continue

        if token.dep_ not in STRONG_DEPS:
            if debug:
                print(f"Skip DEP: {token.text} ({token.dep_})")
            continue

        key = (token.lemma_.lower(), token.dep_, token.head.text)
        if key in seen:
            continue
        seen.add(key)

        particle = next((c.text for c in token.children if c.dep_ == 'prt'), None)
        full_text = f"{token.text} {particle}" if particle else token.text

        modifiers = []
        if any(c.dep_ == 'neg' for c in token.children):
            modifiers.append('negation')
        if any(c.dep_ == 'aux' for c in token.children):
            modifiers.append('auxiliary')

        if token.lemma_ in LIGHT_VERBS:
            for child in token.children:
                if child.dep_ in {'dobj', 'attr'} and child.pos_ == 'NOUN':
                    phrase = f"{token.text} {child.text}"
                    key2 = (phrase.lower(), token.dep_)
                    if key2 not in seen:
                        seen.add(key2)
                        predicates.append({
                            'text': phrase,
                            'lemma': f"{token.lemma_} {child.lemma_}",
                            'pos': 'VERB+NOUN',
                            'dep': token.dep_,
                            'head': token.head.text,
                            'is_predicate': True,
                            'modifiers': modifiers.copy(),
                            'clause_type': token.dep_ if token.dep_ in {'relcl','advcl','xcomp','ccomp'} else None
                        })

        predicates.append({
            'text': full_text,
            'lemma': token.lemma_.lower(),
            'pos': token.pos_,
            'dep': token.dep_,
            'head': token.head.text,
            'is_predicate': True,
            'modifiers': modifiers,
            'clause_type': token.dep_ if token.dep_ in {'relcl','advcl','xcomp','ccomp'} else None
        })

    for token in doc:
        if token.dep_ == 'conj' and token.pos_ in {'VERB','NOUN','ADJ'}:
            key = (token.lemma_.lower(), token.dep_, token.head.text)
            if key in seen:
                continue
            seen.add(key)
            particle = next((c.text for c in token.children if c.dep_=='prt'), None)
            full_text = f"{token.text} {particle}" if particle else token.text
            modifiers = []
            if any(c.dep_=='neg' for c in token.children): modifiers.append('negation')
            if any(c.dep_=='aux' for c in token.children): modifiers.append('auxiliary')
            predicates.append({
                'text': full_text,
                'lemma': token.lemma_.lower(),
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text,
                'is_predicate': True,
                'modifiers': modifiers,
                'clause_type': None
            })
    return predicates

embedding_cache = {}

def get_embedding(text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    tokens = [t.lemma_ for t in nlp(text.lower()) if not t.is_stop]
    normalized = " ".join(tokens)
    if normalized in embedding_cache:
        return embedding_cache[normalized]
    emb = embedding_model.encode(normalized)
    embedding_cache[text] = emb
    embedding_cache[normalized] = emb
    return emb

@lru_cache(maxsize=10000, typed=True)
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word, pos=wn.NOUN)[:3]:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word:
                synonyms.add(name)
            if len(synonyms) >= 10:
                break
        if len(synonyms) >= 10:
            break
    manual = manual_synonyms.get(word.lower(), [])
    synonyms.update(m.lower() for m in manual)
    return list(synonyms)[:10]

def get_np_modifiers(span):
    return [t.lemma_.lower() for t in span if t.pos_ == 'ADJ' or t.dep_ == 'amod']

def prepare_environment(environment_list):
    processed = []
    for phrase in environment_list:
        phrase_doc = nlp(phrase)
        lemmas = tuple(t.lemma_.lower() for t in phrase_doc if t.pos_ in {"NOUN", "PROPN", "ADJ"} and not t.is_stop)
        modifiers = get_np_modifiers(phrase_doc)
        embedding = get_embedding(phrase)
        processed.append({
            'original': phrase,
            'lower': phrase.lower(),
            'lemmas': lemmas,
            'modifiers': modifiers,
            'embedding': embedding
        })
    return processed

def normalize_np_text(text):
    return re.sub(r'^(the|a|an)\s+', '', text.lower())

def find_matches(np_text, np_lemmas, np_embedding, env_processed):
    matches = set()
    match_sources = {}

    np_text_norm = normalize_np_text(np_text)

    env_embeddings = np.stack([e['embedding'] for e in env_processed])

    # 1. Full text match
    if matching_config['use_text_match']:
        for env in env_processed:
            env_text_norm = normalize_np_text(env['lower'])
            if np_text_norm == env_text_norm:
                matches.add(env['original'])
                match_sources[env['original']] = 'text_match'

    # 2. Lemma overlap
    if matching_config['use_lemma_overlap']:
        for env in env_processed:
            if set(np_lemmas) & set(env['lemmas']):
                sim = cosine_similarity([np_embedding], [env['embedding']])[0][0]
                if sim > 0.6: 
                    matches.add(env['original'])
                    match_sources[env['original']] = 'lemma_overlap'

    # 3. Embedding match
    if matching_config['use_embedding']:
        scores = cosine_similarity([np_embedding], env_embeddings)[0]
        for i, score in enumerate(scores):
            env = env_processed[i]
            threshold = max(0.4, matching_config['embedding_threshold_base'] - matching_config['embedding_threshold_length_factor'] * len(env['original'].split()))
            if score > threshold:
                matches.add(env['original'])
                match_sources[env['original']] = 'embedding'

    # 4. Synonym embedding match
    if matching_config['use_wordnet']:
        synonyms = set()
        for lemma in np_lemmas:
            synonyms.update(get_synonyms(lemma))
        for synonym in synonyms:
            syn_emb = get_embedding(synonym)
            scores = cosine_similarity([syn_emb], env_embeddings)[0]
            for i, score in enumerate(scores):
                if score > matching_config['synonym_threshold']:
                    env = env_processed[i]
                    matches.add(env['original'])
                    match_sources[env['original']] = 'synonym_embedding'

    # 5. Manual synonym match
    if matching_config['use_manual']:
        manual = manual_synonyms.get(np_text, [])
        for m in manual:
            for env in env_processed:
                if normalize_np_text(m) == normalize_np_text(env['lower']):
                    matches.add(env['original'])
                    match_sources[env['original']] = 'manual'

    return list(matches), match_sources


def extract_actants_with_embeddings(instruction_text, predicates, environment_list, doc=None, debug=False):
    if doc is None:
        doc = nlp(instruction_text)

    result = {}
    visited = set()
    env_processed = prepare_environment(environment_list)

    def get_np_lemma_tuple(span):
        return tuple(token.lemma_.lower() for token in span if not token.is_stop)

    modifiers_tokens = set()

    for pred in predicates:
        pred_token = next((t for t in doc if t.lemma_.lower() == pred['lemma'].lower() and t.pos_ in {'VERB', 'ADJ'}), None)
        if not pred_token:
            continue

        actants = []

        for chunk in doc.noun_chunks:
            if pred_token not in chunk.root.ancestors and chunk.root.head != pred_token:
                continue

            prep = None
            pobj = None

            for child in chunk.root.children:
                if child.dep_ == 'prep':
                    prep = child
                    for pobj_candidate in child.children:
                        if pobj_candidate.dep_ == 'pobj':
                            pobj = pobj_candidate
                            break

            if prep and pobj:
                end = max(chunk.end, pobj.i + 1)
                extended_span = doc[chunk.start:end]
            else:
                extended_span = chunk

            full_np_text = extended_span.text.lower().strip()
            full_np_lemmas = get_np_lemma_tuple(extended_span)
            full_np_embedding = get_embedding(full_np_text)

            modifiers = [token for token in extended_span if token.dep_ in {'amod', 'compound', 'det', 'nummod'} and token != extended_span.root]
            for mod_token in modifiers:
                modifiers_tokens.add(mod_token.i)

            visited.update(range(extended_span.start, extended_span.end))

            explicit_env_match = [
                e for e in environment_list
                if normalize_np_text(e) in normalize_np_text(full_np_text)
            ]

            if explicit_env_match:
                matches = explicit_env_match
                match_sources = {explicit_env_match[0]: 'explicit_modifier'}
                is_ambiguous = False
            else:
                exact_matches = [e for e in environment_list if normalize_np_text(e) == normalize_np_text(full_np_text)]

                if exact_matches:
                    matches = exact_matches
                    match_sources = {exact_matches[0]: 'exact_match'}
                    is_ambiguous = False
                else:
                    matches, match_sources = find_matches(full_np_text, full_np_lemmas, full_np_embedding, env_processed)
                    is_ambiguous = len(matches) > 1

            confidence = max([weight_map.get(src, 0.5) for src in match_sources.values()], default=0)

            actants.append({
                'text': extended_span.text,
                'lemma': full_np_text,
                'dep': extended_span.root.dep_,
                'matches': matches,
                'match_sources': match_sources,
                'modifiers': [t.text for t in modifiers],
                'is_ambiguous': is_ambiguous,
                'is_in_environment': bool(matches),
                'confidence': confidence,
                'is_full_np_match': True,
                'range': (extended_span.start_char, extended_span.end_char),
            })

            if debug:
                print(f"NP match: {extended_span.text} → {matches} | modifiers: {[t.text for t in modifiers]}")

        for token in doc:
            if token.i in visited or token.is_stop or token.pos_ not in {'NOUN', 'PROPN', 'ADJ'}:
                continue
            if token.dep_ in {'amod', 'compound', 'det', 'nummod'} or token.i in modifiers_tokens:
                continue

            lemma = token.lemma_.lower()
            token_emb = get_embedding(lemma)
            matches, match_sources = find_matches(lemma, [lemma], token_emb, env_processed)
            confidence = max([weight_map.get(src, 0.5) for src in match_sources.values()], default=0)

            if matches:
                actants.append({
                    'text': token.text,
                    'lemma': lemma,
                    'dep': token.dep_,
                    'matches': matches,
                    'match_sources': match_sources,
                    'modifiers': [],
                    'is_ambiguous': len(matches) > 1,
                    'is_in_environment': True,
                    'confidence': confidence,
                    'is_full_np_match': False,
                    'range': (token.idx, token.idx + len(token.text)),
                })

                visited.add(token.i)

                if debug:
                    print(f"Token match: {token.text} → {matches}")

        filtered_actants = []
        for a in actants:
            if not a.get("matches"):
                continue
            a_range = a.get("range")
            a_matches = set(a["matches"])
            is_subsumed = any(
                b != a
                and b.get("range")
                and a_range[0] >= b["range"][0]
                and a_range[1] <= b["range"][1]
                and set(b["matches"]) >= a_matches
                for b in actants
            )
            if not is_subsumed:
                filtered_actants.append(a)

        result[pred['lemma'].lower()] = filtered_actants

    return result

def _svc_prompt(kind, **kwargs):
    """Picks the right per-model prompt file for SVC's actant/frame
    templates (prompts/prompts_svc/<family>/<kind>_prompt.txt) — family
    is auto-detected inside response_parser.load_prompt() from
    model.model_name, so nothing needs figuring out here."""
    return response_parser.load_prompt("svc", model.model_name, kind, **kwargs)


def _actant_key(actant):
    """The one string an actant is known by, everywhere.

    v1 had two: build_variants_from_matches() keyed by the bare
    `a['lemma']`, while get_variants_from_llm() sent
    `modifiers + lemma` to the model. The merge in
    get_combined_variants_from_actants() compared those keys, never
    matched, and left the same actant sitting in the dict twice under
    two names — which the old `variability` branch then counted twice.
    """
    lemma = str(actant.get("lemma", "")).strip()
    mods = [str(m).strip() for m in (actant.get("modifiers") or []) if str(m).strip()]
    return (" ".join(mods) + " " + lemma).strip() if mods else lemma


def iter_actants(actants_dict):
    """(predicate, actant) pairs, in a stable order."""
    for pred, actants in actants_dict.items():
        for a in actants:
            yield pred, a


def build_variants_from_matches(actants_dict):
    """Rule-based proposal. Now runs each actant through the specificity
    gate: an actant whose own modifiers already pick one environment
    object is RESOLVED, not expanded.

    This is the single biggest false-positive source in v1. In 63% of
    AmbiK's unambiguous_direct rows the instruction names a multi-word
    environment object verbatim AND that object has same-head siblings
    ("ceramic dinner plate" alongside eight other plates) — head-noun
    matching turned those siblings into variants and the old
    `variability` branch flagged the row.
    """
    variants = {}
    resolved = {}
    for _, a in iter_actants(actants_dict):
        if not a['is_in_environment']:
            continue
        key = _actant_key(a)
        matches = a.get('matches') or []

        if SPECIFICITY_GATE:
            picked, remaining = svc_v2.resolve_by_specificity(key, matches)
        else:
            picked, remaining = (None, matches)

        if picked is not None:
            resolved[key] = picked
            variants[key] = "no variations"
        elif len(remaining) > 1:
            variants[key] = remaining
        else:
            resolved[key] = remaining[0] if remaining else None
            variants[key] = "no variations"

    return variants, resolved


def get_variants_from_llm(actants_dict, environment_list, instruction,
                          manual_synonyms=None, use_llm=True):
    """LLM proposal. Expects actant_prompt.txt's v2 shape: a JSON ARRAY of
    {"actant", "variants", "resolved"}.

    The array matters. `parse_actant_response` returns `[data]` for a
    JSON *object*, and `group_variants_by_np` keeps only entries carrying
    an `np_text`/`actant` key — so v1's object-shaped response parsed to
    {} and the LLM contributed nothing at all to the candidate pool. An
    array whose elements carry "actant" goes through the existing parser
    untouched.
    """
    if manual_synonyms is None:
        manual_synonyms = {}

    actant_entries = [_actant_key(a) for _, a in iter_actants(actants_dict)
                      if a["is_in_environment"]]
    actant_entries = [e for e in actant_entries if e]

    if not use_llm or not actant_entries:
        return {}, {}

    prompt = _svc_prompt(
        "actant",
        actants=", ".join(sorted(set(actant_entries))),
        instruction=instruction,
        environment=", ".join(environment_list),
    )

    if USE_MANUAL_SYNONYMS_IN_LLM_PROMPT and manual_synonyms:
        present = {obj.lower() for obj in environment_list}
        hints = [f"{g}: {', '.join(h)}"
                 for g, inst in manual_synonyms.items()
                 for h in [[i for i in inst if i.lower() in present]] if len(h) > 1]
        if hints:
            prompt += ("\n\nKnown generic-to-instance mappings for this "
                       "environment:\n" + "\n".join(hints))

    try:
        response_text = model.generate(prompt).strip()
        blocks = response_parser.parse_actant_response(response_text)
    except Exception as e:
        print("LLM actant stage failed, rule-based proposal only:", e)
        return {}, {}

    variants, resolved = {}, {}
    for entry in blocks:
        if not isinstance(entry, dict):
            continue
        key = entry.get("actant") or entry.get("np_text")
        if not key:
            continue
        cands = entry.get("variants") or []
        cands = [c for c in cands if isinstance(c, str) and c.strip()]
        if len(cands) > 1:
            variants[key] = cands
        else:
            variants[key] = "no variations"
        if isinstance(entry.get("resolved"), str) and entry["resolved"].strip():
            resolved[key] = entry["resolved"].strip()

    return variants, resolved


def get_combined_variants_from_actants(actants_dict, environment_list, instruction,
                                       use_llm=True, manual_synonyms=None):
    if manual_synonyms is None:
        manual_synonyms = {}

    rule_variants, rule_resolved = build_variants_from_matches(actants_dict)
    combined = dict(rule_variants)
    resolved = dict(rule_resolved)
    match_sources = {k: (['rule'] if v != "no variations" else [])
                     for k, v in rule_variants.items()}

    if use_llm:
        llm_variants, llm_resolved = get_variants_from_llm(
            actants_dict, environment_list, instruction,
            manual_synonyms=manual_synonyms, use_llm=True,
        )
        for k, v in llm_variants.items():
            if v == "no variations":
                continue
            if SPECIFICITY_GATE and k in resolved and resolved[k]:
                continue
            if k not in combined or combined[k] == "no variations":
                combined[k] = list(v)
                match_sources[k] = ['llm']
            else:
                merged = list(dict.fromkeys(list(combined[k]) + list(v)))
                combined[k] = merged
                match_sources.setdefault(k, []).append('llm')

        for k, v in llm_resolved.items():
            resolved.setdefault(k, v)

    for k, v in combined.items():
        if v == []:
            combined[k] = "no variations"

    return combined, match_sources, resolved


def update_ambiguity_flags(actants_dict):
    for pred, actants in actants_dict.items():
        for actant in actants:
            matches = actant.get("matches", [])
            actant["is_ambiguous"] = len(matches) > 1

def run_valency_check(actants_dict, variants, instruction, environment_list):
    """Channel A's semantics: for each (predicate, actant) with an open
    choice, ask what the PREDICATE admits.

    Iterates actants_dict by predicate — v1's activate_frames_llm_by_np
    walked the NP-keyed variants dict instead, which is why the predicate
    was unavailable and frame_prompt.txt never received one. A method
    called Semantic Valency Conflict deciding its central question with
    no access to the valency-bearing element was the core design fault,
    not a prompt-wording issue.
    """
    entries_per_actant = {}

    for pred, a in iter_actants(actants_dict):
        if not a['is_in_environment']:
            continue
        key = _actant_key(a)
        cands = variants.get(key)
        if cands == "no variations" or not cands or len(cands) < 2:
            continue
        if (pred, key) in entries_per_actant:
            continue

        prompt = _svc_prompt(
            "valency",
            predicate=pred,
            lemma=key,
            instruction=instruction,
            variants=", ".join(cands),
            environment=", ".join(environment_list),
        )
        try:
            response_text = model.generate(prompt).strip()
            entries_per_actant[(pred, key)] = response_parser.parse_frame_response(response_text)
        except Exception as e:
            print(f"Error in valency check for '{key}' under '{pred}':", e)
            entries_per_actant[(pred, key)] = []

    return entries_per_actant


def run_hazard_check(actants_dict, resolved, instruction, environment_list):
    """Channel B on RESOLVED objects — the part v1 could not see at all.

    81% of AmbiK's safety rows are not attribute-choice rows: the object
    is named outright and the action on it is still unsafe ("heat the
    stainless steel plate in the microwave"). Any NP-driven pipeline
    scores those at HR=0 no matter how its thresholds are tuned, which is
    why safety sat at 0.450 — the lowest of all four categories, in the
    one category the method is motivated by.
    """
    hits = {}
    if not RUN_HAZARD_CHECK:
        return hits

    calls = 0
    for pred, a in iter_actants(actants_dict):
        if not a['is_in_environment']:
            continue
        key = _actant_key(a)
        filler = resolved.get(key)
        if not filler or (pred, key) in hits:
            continue
        if calls >= HAZARD_CALLS_PER_ROW:
            break
        calls += 1

        prompt = _svc_prompt(
            "hazard", predicate=pred, filler=filler,
            instruction=instruction, environment=", ".join(environment_list),
        )
        try:
            blocks = response_parser.parse_frame_response(model.generate(prompt).strip())
        except Exception as e:
            print(f"Error in hazard check for '{filler}':", e)
            continue
        parsed = blocks[0] if isinstance(blocks, list) and blocks else blocks

        if isinstance(parsed, dict) and parsed.get("hazard") in (True, "true", "True"):
            hits[(pred, key)] = {
                "filler": filler,
                "safer_alternative": parsed.get("safer_alternative"),
                "reason": parsed.get("reason", ""),
            }

    return hits


def run_adjunct_check(predicates, instruction, environment_list):
    """Channel C, narrowed. Kept rather than deleted because it is the
    only channel that reaches safety cases with no object choice at all
    ("cut the unwashed vegetables"). Narrowed by the consequence
    requirement in adjunct_prompt.txt so it stops firing on every kitchen
    instruction with an open Manner slot, and logged as its own channel
    so its contribution is ablatable from any run's raw log.
    """
    hits = []
    if not RUN_ADJUNCT_CHECK:
        return hits

    for pred in list(predicates)[:ADJUNCT_CALLS_PER_ROW]:
        prompt = _svc_prompt("adjunct", predicate=pred, instruction=instruction,
                             environment=", ".join(environment_list))
        try:
            blocks = response_parser.parse_frame_response(model.generate(prompt).strip())
        except Exception as e:
            print(f"Error in adjunct check for '{pred}':", e)
            continue
        block = blocks[0] if isinstance(blocks, list) and blocks else blocks
        if isinstance(block, dict) and block.get("has_adjunct_ambiguity") in (True, "true", "True"):
            hits.append(block)

    return hits


def build_decisions(actants_dict, variants, resolved, valency_entries, hazard_hits):
    """Per-actant decisions, ready for svc_v2.decide_row."""
    decisions = []
    seen = set()

    for pred, a in iter_actants(actants_dict):
        if not a['is_in_environment']:
            continue
        key = _actant_key(a)
        if (pred, key) in seen:
            continue
        seen.add((pred, key))

        if (pred, key) in hazard_hits:
            hit = hazard_hits[(pred, key)]
            decisions.append(svc_v2.ActantDecision(
                actant=key, predicate=pred, ask=True, channel="hazard",
                reason=hit.get("reason") or "predicate acts unsafely on this filler",
                candidates=[c for c in [hit.get("filler"), hit.get("safer_alternative")] if c],
                resolved=resolved.get(key),
            ))
            continue

        entries = valency_entries.get((pred, key))
        cands = variants.get(key)
        if not entries:
            decisions.append(svc_v2.ActantDecision(
                actant=key, predicate=pred, ask=False, channel=None,
                reason="no valency judgement available",
                candidates=[] if cands == "no variations" else list(cands or []),
                resolved=resolved.get(key),
            ))
            continue

        decisions.append(svc_v2.decide_actant(key, pred, entries,
                                              resolved=resolved.get(key)))

    return decisions


def is_intent_aligned_cosine(intents, variants, embedding_model, threshold=0.6):
    if not intents or not variants:
        return False

    intent_embeddings = embedding_model.encode(intents)
    variant_embeddings = embedding_model.encode(list(variants))

    sims = cosine_similarity(intent_embeddings, variant_embeddings)
    max_sim = sims.max() if sims.size > 0 else 0
    return max_sim >= threshold

_VARIANT_SURFACE_KEYS = ("variant", "text")


def extract_strings(value):
    """Flatten a variants entry into a list of lowercased candidate strings.

    Handles BOTH shapes that reach it:
      * rule-based  -> list[str]                      (build_variants_from_matches)
      * LLM-based   -> list[dict]                     (parser.group_variants_by_np)

    See EXTRACT_STRINGS_INCLUDE_LLM_VARIANTS for the legacy behavior.
    """
    if isinstance(value, str):
        return [value.lower()]

    if isinstance(value, list):
        if not EXTRACT_STRINGS_INCLUDE_LLM_VARIANTS:
            # legacy: dict-shaped entries fall out here, silently
            return [v.lower() for v in value if isinstance(v, str)]
        result = []
        for v in value:
            result.extend(extract_strings(v))
        return result

    if isinstance(value, dict):
        if EXTRACT_STRINGS_INCLUDE_LLM_VARIANTS:
            for key in _VARIANT_SURFACE_KEYS:
                sub = value.get(key)
                if isinstance(sub, str) and sub.strip():
                    return [sub.lower()]
        result = []
        for subval in value.values():
            result.extend(extract_strings(subval))
        return result

    return []

def run_disambiguation_pipeline(instruction_text, environment_list, use_llm=True, debug=False):
    # 1
    doc = preprocess_text(instruction_text)
    if debug:
        print("Doc tokens:", [token.text for token in doc])
    
    # 2
    predicates = identify_predicates_from_doc(doc, debug=debug)
    if debug:
        print("Predicates found:", predicates)
    
    # 3
    actants_dict = extract_actants_with_embeddings(instruction_text, predicates, environment_list, doc=doc)
    if debug:
        for pred, actants in actants_dict.items():
            print(f"Actants for predicate '{pred}':")
            for a in actants:
                print(" ", a)
    
    # 4
    update_ambiguity_flags(actants_dict)
    
    # 5 — candidate proposal (matching PROPOSES, valency DISPOSES)
    variants, match_sources, resolved = get_combined_variants_from_actants(
        actants_dict,
        environment_list,
        instruction_text,
        use_llm=use_llm,
        manual_synonyms=manual_synonyms,
    )
    if debug:
        print("Variants:", variants)
        print("Resolved by specificity:", resolved)

    # 6 — channel A: predicate-conditioned valency
    valency_entries = run_valency_check(actants_dict, variants,
                                        instruction_text, environment_list)

    # 7 — channel B: hazard on already-resolved fillers
    hazard_hits = run_hazard_check(actants_dict, resolved,
                                   instruction_text, environment_list)

    # 8 — channel C: peripheral frame elements with a physical consequence
    adjunct_hits = run_adjunct_check(predicates, instruction_text, environment_list)

    # 9 — decision
    decisions = build_decisions(actants_dict, variants, resolved,
                                valency_entries, hazard_hits)
    hr, hr_source, ask_candidates, hr_reason = svc_v2.decide_row(decisions, adjunct_hits)

    if debug:
        for d in decisions:
            print(f"  [{d.predicate}] {d.actant}: ask={d.ask} "
                  f"channel={d.channel} ({d.reason})")
        print(f"HR={hr} via {hr_source}: {hr_reason}")

    # everything the pipeline considered, for SR
    all_candidates = []
    for v in variants.values():
        if v == "no variations":
            continue
        all_candidates.extend(extract_strings(v))
    all_candidates.extend([r for r in resolved.values() if r])
    for hit in hazard_hits.values():
        all_candidates.extend([c for c in [hit.get("filler"),
                                           hit.get("safer_alternative")] if c])

    return {
        "predicates": predicates,
        "actants": actants_dict,
        "variants": variants,
        "resolved": resolved,
        "match_sources": match_sources,
        "valency": {f"{p}|{k}": v for (p, k), v in valency_entries.items()},
        "hazards": {f"{p}|{k}": v for (p, k), v in hazard_hits.items()},
        "adjunct": adjunct_hits,
        "decisions": decisions,
        "hr": hr,
        "hr_source": hr_source,
        "hr_reason": hr_reason,
        "ask_candidates": ask_candidates,
        "all_candidates": all_candidates,
    }

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_llm_module.__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

ambik_dataset = pd.read_csv(os.path.join(_DATA_DIR, "ambik_test.csv"))
ambik_dataset = ambik_dataset[['ambiguity_type', 'environment_short', 'unambiguous_direct', 'ambiguous_task', 'user_intent']]
amb = ambik_dataset[['environment_short', 'ambiguity_type', 'ambiguous_task', 'user_intent']]
ambik_dataset.ambiguity_type = ['unambiguous_direct']*len(ambik_dataset)
ambik_dataset = pd.concat([ambik_dataset, amb])

DATASET_NAME = "ambik"
CHECKPOINT_EVERY = 50 

existing_results, already_done = metrics.load_checkpoint("svc", MODEL_NAME, DATASET_NAME)
if already_done > 0:
    print(f"Found checkpoint with {already_done} rows already done — resuming from there.")

def process_dataset(df, checkpoint_every=CHECKPOINT_EVERY, results=None, skip=0):
    results = list(results) if results else []
    since_checkpoint = 0

    df_remaining = df.iloc[skip:]
    for idx, row in tqdm(df_remaining.iterrows(), total=len(df_remaining)):
        try:
            if row['ambiguity_type'] == 'unambiguous_direct':
                instruction = row['unambiguous_direct']
            else:
                instruction = row['ambiguous_task']

            environment = [x.strip() for x in row['environment_short'].split(',')]

            detection_result = run_disambiguation_pipeline(instruction, environment)

            hr = int(detection_result.get('hr', 0))
            hr_source = detection_result.get('hr_source')
            hr_reason = detection_result.get('hr_reason', '')
            amb_type = row['ambiguity_type']

            chr_value = metrics.correct_help_rate_from_label(hr, amb_type, metrics.AMBIK_HELP_EXPECTED)

            user_intent_raw = row['user_intent'] if pd.notna(row['user_intent']) else ''
            user_intents = re.split(r'[\|,\-\/&]', user_intent_raw.lower())
            user_intents = [x.strip() for x in user_intents if x.strip()]

            found_variants = set(detection_result.get('ask_candidates') or [])

            all_candidates = set(detection_result.get('all_candidates') or [])
            sr = metrics.success_rate(list(all_candidates), user_intents)

            def _cosine_aligned(intents, candidates, _model=embedding_model):
                return is_intent_aligned_cosine(intents, candidates, _model)

            ia = metrics.intent_alignment(hr, found_variants, user_intents, _cosine_aligned)

            results.append({
                'ambiguity_type': amb_type,
                'HR': hr,
                'CHR': chr_value,
                'IA': ia,
                'SR': sr,
                'hr_source': hr_source or '',
                'hr_reason': hr_reason,
            })
            since_checkpoint += 1

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

        if since_checkpoint >= checkpoint_every:
            metrics.save_metrics(
                results, "svc", MODEL_NAME, DATASET_NAME, family="binary",
                all_categories=metrics.AMBIK_CATEGORIES,
                help_expected_types=metrics.AMBIK_HELP_EXPECTED,
                checkpoint=True,
            )
            since_checkpoint = 0
            print(f"  [checkpoint] saved {len(results)} rows so far -> "
                  f"svc_{MODEL_NAME}_{DATASET_NAME}/")

    metrics.save_metrics(
        results, "svc", MODEL_NAME, DATASET_NAME, family="binary",
        all_categories=metrics.AMBIK_CATEGORIES,
        help_expected_types=metrics.AMBIK_HELP_EXPECTED,
        checkpoint=False,
    )
    return pd.DataFrame(results)


# Smoke test: only the first 10 rows. Flip TEST_MODE = False (or just call process_dataset(ambik_dataset) directly)
TEST_MODE = _cli_args.svc_limit is not None
test_df = ambik_dataset.head(_cli_args.svc_limit or 10) if TEST_MODE else ambik_dataset

results_df = process_dataset(test_df, results=existing_results, skip=already_done)

metrics.print_final_report(
    results_df, "svc", MODEL_NAME, DATASET_NAME, family="binary",
    all_categories=metrics.AMBIK_CATEGORIES,
    help_expected_types=metrics.AMBIK_HELP_EXPECTED,
)