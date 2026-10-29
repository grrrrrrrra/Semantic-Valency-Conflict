import gc
import os
import numpy as np
import pandas as pd
from scipy import spatial
from gensim.models import FastText
from gensim.test.utils import common_texts
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm import load_llm
import llm as _llm_module
from pipeline import LoFreePipe


def get_logits(lofree, sample):
    """Calculate various metrics and scores for a given sample.

    Args:
        lofree: LoFreePipe instance
        sample: Dictionary containing sample data

    Returns:
        Processed sample with calculated metrics
    """
    sample = lofree.generate_options(sample)
    gc.collect()
    sample = lofree.frequency(sample)
    normalized_entropy = lofree.normalized_entropy(sample['frequences'])
    sample = lofree.semantic_similarity(sample)
    sample = lofree.nonconformity_score(sample, normalized_entropy)
    gc.collect()
    return sample


def filter_similar_sentences(A, intents, embedding_model, threshold=0.6):
    """Filter sentences based on similarity to target intents.

    Was substring containment; now cosine similarity via embedding_model
    — same is_intent_aligned_cosine mechanism used everywhere else now
    (SVC, and LoFreeCP's own IA at eval time). Matters here specifically:
    conformal prediction's coverage guarantee is only meaningful if
    calibration's notion of "the true label's score" agrees with
    evaluation's notion of "correct" — substring here vs cosine at eval
    time would be two different definitions of match inside the same
    pipeline. NOTE: this compares a short intent label against a full
    generated sentence (not two short labels, like SVC's comparison) —
    the 0.6 threshold was not tuned for that specific comparison shape;
    sanity-check calibration output before trusting it blindly.

    Args:
        A: Dictionary of {sentence: nonconformity_score}
        intents: List of target intents to match against
        embedding_model: SentenceTransformer instance
        threshold: cosine similarity cutoff (same default as
            is_intent_aligned_cosine elsewhere — see the NOTE above)

    Returns:
        Filtered dictionary containing only sentences matching any intent
    """
    if not intents or not A:
        return {}
    keys = list(A.keys())
    intent_embeddings = embedding_model.encode(intents)
    answer_embeddings = embedding_model.encode(keys)
    sims = cosine_similarity(intent_embeddings, answer_embeddings)
    return {k: A[k] for i, k in enumerate(keys) if sims[:, i].max() >= threshold}


def calibtation(calib_path, lofree, embedding_model):
    """Calculate calibration threshold for conformal prediction.

    Args:
        calib_path: Path to calibration data CSV
        lofree: an already-constructed LoFreePipe (model loaded once by
            the caller, same as the runner does) — was previously built
            fresh inside this function from lofree.yaml on every call.
        embedding_model: SentenceTransformer instance, for
            filter_similar_sentences' cosine-similarity intent matching.

    Returns:
        Calculated quantile threshold (qhat)
    """
    target_success = 0.8
    epsilon = 1 - target_success

    calib_data = pd.read_csv(calib_path)

    calibration_scores = []
    for _, row in calib_data.iterrows():
        sample = {
            'id': row['id'],
            'description': row['environment_full'],
            'task': row['task'],
            'intents': row['user_intent'].split(',') if isinstance(row['user_intent'], str) else []
        }

        try:
            sample = get_logits(lofree, sample)
            filtered = filter_similar_sentences(
                sample['nonconformity_scores'],
                sample['intents'],
                embedding_model,
            )
            calibration_scores.extend(filtered.values())
        except Exception as e:
            print(f"Error processing sample {row['id']}: {str(e)}")
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
    lofree = LoFreePipe(model)
    embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_llm_module.__file__)))
    _DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
    CALIB_PATH = os.path.join(_DATA_DIR, "introplan_calib.csv")

    threshold = calibtation(CALIB_PATH, lofree, embedding_model)
    print(f"Calibration threshold (IntroPlan, {MODEL_NAME}): {threshold}")

    pd.DataFrame({'threshold': [threshold]}).to_csv("calibration_params_introplan.csv", index=False)