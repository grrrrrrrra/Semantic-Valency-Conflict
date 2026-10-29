import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm import load_llm
import llm as _llm_module  # only used to locate utilities/ below — path resolution, not model loading
from pipeline import AffordancesPipe
from objects_ambik import environment_objects  # assumed to sit alongside this script


def is_intent_aligned_cosine(intents, variants, embedding_model, threshold=0.6):
    if not intents or not variants:
        return False
    intent_embeddings = embedding_model.encode(intents)
    variant_embeddings = embedding_model.encode(list(variants))
    sims = cosine_similarity(intent_embeddings, variant_embeddings)
    max_sim = sims.max() if sims.size > 0 else 0
    return max_sim >= threshold


def get_affordance_scores(affordances, description, task):
    """Runs LAP's full options -> choosing/affordance pipeline (minus the
    final CP threshold) and returns the option dict plus the raw
    (pre-threshold) affordance scores."""
    options = affordances.predict_examples(description, task, prefix='', action=task)
    _, affordance_scores, _ = affordances.generate_answer(description, task, '', task, options)
    return options, affordance_scores


def calibtation(calib_path, affordances, embedding_model):
    """Calculate calibration threshold for conformal prediction.

    Args:
        calib_path: Path to calibration data CSV
        affordances: an already-constructed AffordancesPipe (model loaded
            once by the caller, same as the runner does)
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

            options, affordance_scores = get_affordance_scores(affordances, description, task)
            for letter, score in affordance_scores.items():
                text = affordances._strip_label(letter, options.get(letter, 'do nothing'))
                if is_intent_aligned_cosine(intents, [text], embedding_model):
                    calibration_scores.append(score)
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
    affordances = AffordancesPipe(model, environment_objects)
    embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_llm_module.__file__)))
    _DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
    CALIB_PATH = os.path.join(_DATA_DIR, "ambik_calib.csv")

    threshold = calibtation(CALIB_PATH, affordances, embedding_model)
    print(f"Calibration threshold (LAP, AmbiK, {MODEL_NAME}): {threshold}")

    pd.DataFrame({'threshold': [threshold]}).to_csv("calibration_params_ambik.csv", index=False)