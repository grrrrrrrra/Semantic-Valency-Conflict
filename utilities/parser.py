"""
One parser for model responses across all five methods in this project:
LoFreeCP, KnowNo, LAP (Affordances), Prompting, SVC.

Which pipeline calls what:
  LoFreeCP  -> parse_options_as_list
  KnowNo    -> parse_options_as_dict
  LAP       -> parse_options_as_dict, parse_affordance_response
  Prompting -> parse_prompting_response
  SVC       -> parse_actant_response, parse_frame_response,
               group_variants_by_np, load_prompt
"""
import re
import json
from collections import defaultdict

def _strip_code_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _preclean(text):
    return _strip_code_fences(_strip_think_tags(text))

_DIGIT_TO_LETTER = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
_LETTER_LINE_RE = re.compile(r'^([A-Da-d1-4])[.)]\s*(.+)$')


def _parse_lettered_lines(text):
    found = {}
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        clean_line = re.sub(r'^[\*\-\s]+|[\*\s]+$', '', line)
        m = _LETTER_LINE_RE.match(clean_line)
        if not m:
            continue
        key = _DIGIT_TO_LETTER.get(m.group(1).upper(), m.group(1).upper())
        if key in found:
            continue
        content = m.group(2).strip()
        found[key] = (f"{key}) {content}", content)
    return found


def parse_options_as_dict(text, default="do nothing", letters=('A', 'B', 'C', 'D')):
    found = _parse_lettered_lines(text)
    return {letter: (found[letter][0] if letter in found else default) for letter in letters}


def parse_options_as_list(text):
    found = _parse_lettered_lines(text)
    seen = []
    for _, content in found.values():
        c = content.lower().strip()
        if c not in seen:
            seen.append(c)
    return seen

def parse_affordance_response(text):
    if not text:
        return None
    cleaned = _preclean(text).lower()
    has_true = re.search(r'\btrue\b', cleaned) is not None
    has_false = re.search(r'\bfalse\b', cleaned) is not None
    if has_true and not has_false:
        return True
    if has_false and not has_true:
        return False
    return None 

def _parse_json_or_blocks(text):
    text = _preclean(text)

    try:
        data = json.loads(text)
        return [data] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        pass

    results = []
    current_block = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        variant_match = re.match(r"^-?\s*(.+?):\s*$", line)
        if variant_match:
            if current_block:
                results.append(current_block)
                current_block = {}
            current_block['variant'] = variant_match.group(1).strip().rstrip(':')
            continue

        key_val_match = re.match(r"^([a-zA-Z_]+)\s*:\s*(.+)$", line)
        if key_val_match:
            key, val = key_val_match.group(1).strip(), key_val_match.group(2).strip()
            current_block[key] = [v.strip() for v in val.split(',')] if ',' in val else val

    if current_block:
        results.append(current_block)

    return results


def parse_actant_response(text):
    return _parse_json_or_blocks(text)


def parse_frame_response(text):
    return _parse_json_or_blocks(text)


def group_variants_by_np(parsed_blocks):
    grouped = defaultdict(list)
    for entry in parsed_blocks:
        np_text = entry.get("np_text") or entry.get("actant")
        if np_text:
            grouped[np_text].append(entry)
    return dict(grouped)

def _clean_json_text(text):
    text = re.sub(r"(?<!\\)'", '"', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',(\s*[}\]])', r'\1', text)          # trailing commas
    text = re.sub(r'(\w+)\s*:', r'"\1":', text)          # bare keys

    if text.startswith('{') and text.endswith('}'):
        return text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text


def _reconstruct_json_from_text(text):
    label = "Unambiguous"
    if any(w in text.lower() for w in ['ambiguous', 'multiple', 'several', 'different']):
        label = "Ambiguous"
    elif any(w in text.lower() for w in ['unambiguous', 'clear', 'single', 'specific']):
        label = "Unambiguous"

    ambiguities = []
    if label == "Ambiguous":
        phrase_match = re.search(r'"(.*?)"', text)
        phrase = phrase_match.group(1) if phrase_match else "object"

        candidates = []
        for pattern in [r'"([^"]*)"', r'\[(.*?)\]', r'candidates?:?\s*([^\.\n]+)']:
            for match in re.findall(pattern, text, re.IGNORECASE):
                items = [item.strip().strip('"\'') for item in match.split(',')]
                candidates.extend(item for item in items if len(item) > 1)

        if len(candidates) >= 2:
            ambiguities.append({"phrase": phrase, "candidates": candidates[:2]})

    return json.dumps({"label": label, "ambiguities": ambiguities})


def _fix_json(text):
    label = "Unambiguous"
    label_match = re.search(r'"label"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
    if label_match and 'ambiguous' in label_match.group(1).lower():
        label = "Ambiguous"

    ambiguities = []
    if label == "Ambiguous":
        amb_match = re.search(r'"ambiguities"\s*:\s*\[(.*?)\]', text, re.DOTALL | re.IGNORECASE)
        if amb_match:
            for obj_text in re.findall(r'\{(.*?)\}', amb_match.group(1), re.DOTALL):
                phrase_match = re.search(r'"phrase"\s*:\s*"([^"]*)"', obj_text, re.IGNORECASE)
                phrase = phrase_match.group(1) if phrase_match else "item"

                cand_match = re.search(r'"candidates"\s*:\s*\[(.*?)\]', obj_text, re.DOTALL | re.IGNORECASE)
                candidates = re.findall(r'"([^"]*)"', cand_match.group(1)) if cand_match else []
                candidates = [c for c in candidates if c.strip()]

                if candidates:
                    ambiguities.append({"phrase": phrase, "candidates": candidates})

    return {"label": label, "ambiguities": ambiguities}


def _validate_prompting_result(result):
    if not isinstance(result, dict):
        return {"label": "Unambiguous", "ambiguities": []}

    if 'label' not in result:
        result['label'] = 'Unambiguous'
    elif result['label'] not in ['Ambiguous', 'Unambiguous']:
        label_text = str(result['label']).lower()
        result['label'] = 'Ambiguous' if any(w in label_text for w in ['ambiguous', 'multiple', 'vague']) else 'Unambiguous'

    ambiguities = result.get('ambiguities', [])
    if not isinstance(ambiguities, list):
        ambiguities = []

    valid = []
    for amb in ambiguities:
        if not isinstance(amb, dict):
            continue
        phrase = amb.get('phrase', '')
        if not phrase or phrase in ['the ambiguous word or phrase', 'ambiguous phrase', 'item']:
            phrase = amb['phrase'] if isinstance(amb.get('phrase'), str) and len(amb['phrase']) > 5 else "object"

        candidates = amb.get('candidates', [])
        if not isinstance(candidates, list):
            candidates = []
        filtered = [c for c in candidates if isinstance(c, str) and c not in
                    ['interpretation1', 'interpretation2', 'option1', 'option2']]

        if len(filtered) >= 2:
            valid.append({"phrase": phrase, "candidates": filtered[:2]})

    result['ambiguities'] = valid

    if result['label'] == 'Unambiguous' and result['ambiguities']:
        result['label'] = 'Ambiguous'
    elif result['label'] == 'Ambiguous' and not result['ambiguities']:
        result['label'] = 'Unambiguous'

    return result


def parse_prompting_response(text, original_prompt=""):
    text = text.strip()
    if original_prompt and original_prompt in text:
        text = text.split(original_prompt)[-1].strip()
    text = _preclean(text)

    json_start = text.find('{')
    if json_start == -1:
        if '"label"' in text:
            text = '{' + text[text.find('"label"'):]
        else:
            text = _reconstruct_json_from_text(text)
    else:
        text = text[json_start:]

    json_end = text.rfind('}')
    if json_end != -1:
        text = text[:json_end + 1]

    text = _clean_json_text(text)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = _fix_json(text)

    return _validate_prompting_result(result)

import os
from llm import detect_model_family

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def load_prompt(method, model_name, kind, **placeholders):
    family = detect_model_family(model_name)
    base_dir = os.path.join(PROMPTS_DIR, f"prompts_{method}", family)

    path = None
    for filename in (f"{kind}_prompt.txt", f"{kind}.txt"):
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            path = candidate
            break
    if path is None:
        raise FileNotFoundError(
            f"No prompt file for method={method!r} family={family!r} kind={kind!r} "
            f"— tried '{kind}_prompt.txt' and '{kind}.txt' under {base_dir}"
        )

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if "---SYSTEM---" in text:
        system, user = _split_system_user(text, path)
        return _fill(f"System: {system}\n\nUser: {user}", placeholders)

    return _fill(text, placeholders)


def _split_system_user(text, path):
    sys_match = re.search(r"---SYSTEM---\s*\n(.*?)(?=\n---USER---)", text, re.DOTALL)
    usr_match = re.search(r"---USER---\s*\n(.*)", text, re.DOTALL)
    if not sys_match or not usr_match:
        raise ValueError(f"Expected ---SYSTEM---/---USER--- sections in {path}")
    return sys_match.group(1).strip(), usr_match.group(1).strip()


def _fill(template, placeholders):
    result = template
    for key, value in placeholders.items():
        value_str = "" if value is None else str(value)
        result = result.replace("{{" + key + "}}", value_str)
        result = result.replace("<" + key.upper() + ">", value_str)
    return result