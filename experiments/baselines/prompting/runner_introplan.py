import os
import pandas as pd
from tqdm import tqdm

from llm import load_llm
import llm as _llm_module 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline import PromptingPipe
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

prompting = PromptingPipe(model)

DATASET_NAME = "introplan"
CHECKPOINT_EVERY = 50 


def process_dataset(df, checkpoint_every=CHECKPOINT_EVERY):
    results = []
    since_checkpoint = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            instruction = row['task']
            environment = row['environment_short']
            amb_type = row['ambiguity_type']

            m = prompting.calculate_metrics(
                instruction, environment, row['user_intent'], amb_type,
                metrics.INTROPLAN_HELP_EXPECTED, _cosine_aligned,
            )

            results.append({
                'ambiguity_type': amb_type,
                'HR': m['HR'],
                'CHR': m['CHR'],
                'IA': m['IA'],
                'SR': m['SR'],
            })
            since_checkpoint += 1

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

        if since_checkpoint >= checkpoint_every:
            metrics.save_metrics(
                results, "prompting", MODEL_NAME, DATASET_NAME, family="binary",
                all_categories=metrics.INTROPLAN_CATEGORIES,
                help_expected_types=metrics.INTROPLAN_HELP_EXPECTED,
                checkpoint=True,
            )
            since_checkpoint = 0
            print(f"  [checkpoint] saved {len(results)} rows so far -> "
                  f"prompting_{MODEL_NAME}_{DATASET_NAME}/")
            
    metrics.save_metrics(
        results, "prompting", MODEL_NAME, DATASET_NAME, family="binary",
        all_categories=metrics.INTROPLAN_CATEGORIES,
        help_expected_types=metrics.INTROPLAN_HELP_EXPECTED,
        checkpoint=False,
    )
    return pd.DataFrame(results)


introplan_dataset = pd.read_csv(os.path.join(_DATA_DIR, "introplan_test.csv"))

# Smoke test: only the first 10 rows. Flip TEST_MODE = False
TEST_MODE = False
test_df = introplan_dataset.head(10) if TEST_MODE else introplan_dataset

results_df = process_dataset(test_df)

metrics.print_final_report(
    results_df, "prompting", MODEL_NAME, DATASET_NAME, family="binary",
    all_categories=metrics.INTROPLAN_CATEGORIES,
    help_expected_types=metrics.INTROPLAN_HELP_EXPECTED,
)