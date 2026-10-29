import os
import pandas as pd
from tqdm import tqdm

from llm import load_llm
import llm as _llm_module  
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import AffordancesPipe
from objects_ambik import environment_objects 
import metrics


MODEL_NAME = "gpt-5.6"
model = load_llm(MODEL_NAME)

embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')


def is_intent_aligned_cosine(intents, variants, embedding_model, threshold=0.6):
    if not intents or not variants:
        return False
    intent_embeddings = embedding_model.encode(intents)
    variant_embeddings = embedding_model.encode(list(variants))
    sims = cosine_similarity(intent_embeddings, variant_embeddings)
    max_sim = sims.max() if sims.size > 0 else 0
    return max_sim >= threshold


def _cosine_aligned(intents, candidates, _model=embedding_model):
    return is_intent_aligned_cosine(intents, candidates, _model)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_llm_module.__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_params_ambik.csv")
cp = None
if os.path.exists(_CALIB_PATH):
    cp = pd.read_csv(_CALIB_PATH)['threshold'].iloc[0]
    print(f"Loaded calibrated threshold from {_CALIB_PATH}: {cp}")
else:
    print(f"No calibration file at {_CALIB_PATH} — using AffordancesPipe's hardcoded default cp. "
          f"Run calibrate_ambik.py first to use a calibrated threshold.")

affordances = AffordancesPipe(model, environment_objects, cp=cp)

DATASET_NAME = "ambik"
CHECKPOINT_EVERY = 50

existing_results, already_done = metrics.load_checkpoint("lap", MODEL_NAME, DATASET_NAME)
if already_done > 0:
    print(f"Found checkpoint with {already_done} rows already done — resuming from there.")

def process_dataset(df, checkpoint_every=CHECKPOINT_EVERY, results=None, skip=0):
    results = list(results) if results else []
    since_checkpoint = 0

    df_remaining = df.iloc[skip:]
    for idx, row in tqdm(df_remaining.iterrows(), total=len(df_remaining)):
        try:
            if row['ambiguity_type'] == 'unambiguous_direct':
                task = row['unambiguous_direct']
            else:
                task = row['ambiguous_task']
            description = row['environment_short']
            amb_type = row['ambiguity_type']
            intents = row['user_intent'].split(',') if isinstance(row['user_intent'], str) else []

            option, selected_answers = affordances.run(description, task, prefix='', action=task)

            m = affordances.calculate_metrics(
                option, selected_answers, amb_type, intents,
                metrics.AMBIK_HELP_EXPECTED, _cosine_aligned,
            )

            results.append({
                'ambiguity_type': amb_type,
                'SR': m['SR'],
                'help_rate': m['help_rate'],
                'correct_help_rate': m['correct_help_rate'],
                'IA': m['IA'],
            })
            since_checkpoint += 1

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

        if since_checkpoint >= checkpoint_every:
            metrics.save_metrics(
                results, "lap", MODEL_NAME, DATASET_NAME, family="action_set",
                all_categories=metrics.AMBIK_CATEGORIES,
                help_expected_types=metrics.AMBIK_HELP_EXPECTED,
                checkpoint=True,
            )
            since_checkpoint = 0
            print(f"  [checkpoint] saved {len(results)} rows so far -> "
                  f"lap_{MODEL_NAME}_{DATASET_NAME}/")

    metrics.save_metrics(
        results, "lap", MODEL_NAME, DATASET_NAME, family="action_set",
        all_categories=metrics.AMBIK_CATEGORIES,
        help_expected_types=metrics.AMBIK_HELP_EXPECTED,
        checkpoint=False,
    )
    return pd.DataFrame(results)


ambik_dataset = pd.read_csv(os.path.join(_DATA_DIR, "ambik_test.csv"))
ambik_dataset = ambik_dataset[['ambiguity_type', 'environment_short', 'unambiguous_direct', 'ambiguous_task', 'user_intent']]
amb = ambik_dataset[['environment_short', 'ambiguity_type', 'ambiguous_task', 'user_intent']]
ambik_dataset.ambiguity_type = ['unambiguous_direct'] * len(ambik_dataset)
ambik_dataset = pd.concat([ambik_dataset, amb])

# Smoke test: only the first 10 rows. Flip TEST_MODE = False
TEST_MODE = False
test_df = ambik_dataset.head(10) if TEST_MODE else ambik_dataset

results_df = process_dataset(test_df, results=existing_results, skip=already_done)

metrics.print_final_report(
    results_df, "lap", MODEL_NAME, DATASET_NAME, family="action_set",
    all_categories=metrics.AMBIK_CATEGORIES,
    help_expected_types=metrics.AMBIK_HELP_EXPECTED,
)