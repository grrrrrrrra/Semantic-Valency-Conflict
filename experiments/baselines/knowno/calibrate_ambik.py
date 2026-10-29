import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm import load_llm
import llm as _llm_module  
from pipeline import KnowNoPipe


def is_intent_aligned_cosine(intents, variants, embedding_model, threshold=0.6):
    if not intents or not variants:
        return False
    intent_embeddings = embedding_model.encode(intents)
    variant_embeddings = embedding_model.encode(list(variants))
    sims = cosine_similarity(intent_embeddings, variant_embeddings)
    max_sim = sims.max() if sims.size > 0 else 0
    return max_sim >= threshold


def _normalize_letter(key, mapping_1=('A', 'B', 'C', 'D')):
    if key.isdigit():
        return mapping_1[int(key) - 1]
    return key.upper()


def filter_matching_logits(option, filtered_logits, intents, embedding_model, threshold=0.6):
    """For each candidate token whose corresponding OPTION TEXT
    cosine-matches any true intent, keep its (pre-threshold) logit value
    for the calibration pool — same role as LoFreeCP's
    filter_similar_sentences, adapted to KnowNo's letter/digit-keyed
    logits dict instead of a text->score dict."""
    matched = []
    for key, score in filtered_logits.items():
        letter = _normalize_letter(key)
        if letter not in option:
            continue
        text = option[letter]
        if is_intent_aligned_cosine(intents, [text], embedding_model, threshold):
            matched.append(score)
    return matched


def get_logits(knowno, description, task):
    """Runs KnowNo's two LLM calls (generate options, then score them)
    and returns both the option dict and the raw (pre-threshold) filtered
    logits — calibration needs the logits before self.cp is applied, to
    find qhat in the first place."""
    option = knowno.predict_examples(description, task, prefix='', action=task)
    answer_prompt = knowno.answer_prompt(option, description, task, prefix='', action=task)
    filtered_logits, _ = knowno.generate_answer(answer_prompt)
    return option, filtered_logits


def calibtation(calib_path, knowno, embedding_model):
    """Calculate calibration threshold for conformal prediction.

    Args:
        calib_path: Path to calibration data CSV
        knowno: an already-constructed KnowNoPipe (model loaded once by
            the caller, same as the runner does)
        embedding_model: SentenceTransformer instance, for cosine-based
            intent matching

    Returns:
        Calculated quantile threshold (qhat)
    """
    target_success = 0.8
    epsilon = 1 - target_success

    calib_data = pd.read_csv(calib_path)

    calibration_scores = []
    for idx, row in calib_data.iterrows():
        try:
            if row['ambiguity_type'] == 'unambiguous_direct':
                task = row['unambiguous_direct']
            else:
                task = row['ambiguous_task']
            description = row['environment_short']
            intents = row['user_intent'].split(',') if isinstance(row['user_intent'], str) else []

            option, filtered_logits = get_logits(knowno, description, task)
            matched = filter_matching_logits(option, filtered_logits, intents, embedding_model)
            calibration_scores.extend(matched)
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue

    if calibration_scores:
        q_level = np.ceil((len(calibration_scores) + 1) * (1 - epsilon)) / len(calibration_scores)
        qhat = np.quantile(calibration_scores, q_level)
    else:
        qhat = 0.0

    return qhat


if __name__ == "__main__":
    MODEL_NAME = "gpt-5.6"
    model = load_llm(MODEL_NAME)
    knowno = KnowNoPipe(model)
    embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_llm_module.__file__)))
    _DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
    CALIB_PATH = os.path.join(_DATA_DIR, "ambik_calib.csv")

    threshold = calibtation(CALIB_PATH, knowno, embedding_model)
    print(f"Calibration threshold (KnowNo, AmbiK, {MODEL_NAME}): {threshold}")

    pd.DataFrame({'threshold': [threshold]}).to_csv("calibration_params_ambik.csv", index=False)