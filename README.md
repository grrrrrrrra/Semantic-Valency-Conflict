# SVC: Semantic Valency Conflict

Code for **SVC (Semantic Valency Conflict)**, a method for detecting ambiguity in natural-language instructions given to embodied agents, together with four baselines (KnowNo, LAP, LoFreeCP, Prompting). All methods are evaluated on two datasets: AmbiK and IntroPlan.

For each predicate in an instruction, SVC extracts its actants, matches them against the objects available in the environment, and, when an actant has several candidate objects, asks an LLM whether the predicate's valency admits these candidates interchangeably. If it does not, the instruction is flagged as ambiguous and the agent should ask the user for clarification.

## Repository structure

```
.
├── data/                       # evaluation data (see "Data")
├── experiments/
│   ├── SVC/
│   │   ├── config.yaml         # environment-matching criteria, weights, thresholds, manual synonyms
│   │   ├── svc_v2.py
│   │   ├── runner_ambik.py
│   │   └── runner_introplan.py
│   └── baselines/
│       ├── knowno/             # calibrate_*.py, runner_*.py, pipeline.py
│       ├── lap/                # calibrate_*.py, runner_*.py, pipeline.py, objects_*.py
│       ├── lofreecp/           # calibrate_*.py, runner_*.py, pipeline.py
│       └── prompting/          # runner_*.py, pipeline.py
├── utilities/
│   ├── llm.py                  # model loading and generation (local Hugging Face models and OpenRouter API)
│   ├── llm_config.yaml         # registry of LLM backbones and their generation settings
│   ├── parser.py               # prompt loading and parsing of LLM responses
│   ├── metrics.py              # metrics, aggregation, checkpointing
│   └── prompts/                # prompts_<method>/<model family>/*.txt
└── requirements.txt
```

## Installation

The experiments were run with Python 3.10.

```bash
conda create -n svc python=3.10
conda activate svc
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

The NLTK WordNet corpus and the `paraphrase-mpnet-base-v2` sentence-transformers model are downloaded automatically on the first run.

## LLM backbones

Backbones are registered in `utilities/llm_config.yaml`; scripts refer to them by their key (for example `qwen3-8b`, `mistral-7b`, `gpt-4o`).

- **Local models** (Qwen, Mistral) are loaded from the Hugging Face Hub and require a CUDA GPU.
- **API models** (model names starting with `openai/` or `anthropic/`) are called through [OpenRouter](https://openrouter.ai). Set your key before running:

  ```bash
  export OPENROUTER_API_KEY=<your key>
  ```

Prompts are selected by model family (`qwen`, `mistral`, `gpt`, `claude`), which is detected from the model name; the prompt files are in `utilities/prompts/prompts_<method>/<family>/`.

## Running experiments

All scripts import modules from `utilities/`, so add it to `PYTHONPATH` from the repository root:

```bash
export PYTHONPATH="$(pwd)/utilities"
```

### SVC

The backbone is passed with `--model` (default: `qwen3-32b`, or the value of the `SVC_MODEL` environment variable).

```bash
cd experiments/SVC
python runner_ambik.py --model qwen3-8b              # full AmbiK test set
python runner_ambik.py --model qwen3-8b --limit 10   # smoke test on the first 10 rows
python runner_introplan.py --model qwen3-8b          # IntroPlan test set
```

Pipeline switches (`EXTRACT_STRINGS_INCLUDE_LLM_VARIANTS`, `SPECIFICITY_GATE`, `USE_MANUAL_SYNONYMS_IN_LLM_PROMPT`, `RUN_HAZARD_CHECK`, `RUN_ADJUNCT_CHECK`) are set at the top of each runner. Environment-matching parameters are in `experiments/SVC/config.yaml`.

### Baselines

The backbone is set by `MODEL_NAME` at the top of each script.

KnowNo, LAP and LoFreeCP use conformal prediction and need a calibration step first. Run the calibration script **from the method's directory**: it writes `calibration_params_<dataset>.csv` to the current directory, and the runner reads it from its own directory. If the file is missing, the runner falls back to the pipeline's default threshold. The file name does not include the model, so rerun calibration after changing the backbone.

```bash
cd experiments/baselines/knowno      # or lap, lofreecp
python calibrate_ambik.py
python runner_ambik.py

python calibrate_introplan.py
python runner_introplan.py
```

Prompting needs no calibration:

```bash
cd experiments/baselines/prompting
python runner_ambik.py
python runner_introplan.py
```

KnowNo and LAP use token log-probabilities. For API models, the provider on OpenRouter must return logprobs; otherwise `llm.py` raises an error.

### Outputs

Each run writes its results to `<method>_<model>_<dataset>/` in the current directory:

- `raw_metrics.csv` — per-instruction results;
- `aggregated_metrics.csv` — metrics aggregated per ambiguity category.

For the binary-decision methods, the aggregated columns are `hr_agg` (help rate), `chr_agg` (correct help rate: whether the decision to ask matches the category), `sr_agg` (success rate) and `chq_agg` (harmonic mean of `sr_agg` and `chr_agg`). All metric definitions are in `utilities/metrics.py`.

Results are checkpointed every 50 rows. The runners that call `metrics.load_checkpoint` (SVC, LAP, LoFreeCP, and KnowNo on IntroPlan) resume from an existing checkpoint.

## Data

| File | Rows | Source |
|---|---|---|
| `data/ambik_calib.csv` | 500 | AmbiK |
| `data/ambik_test.csv` | 500 | AmbiK |
| `data/introplan_calib.csv` | 100 | IntroPlan |
| `data/introplan_test.csv` | 500 | IntroPlan |

- **AmbiK** — Ivanova et al., *AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment*, ACL 2025. Original data: https://github.com/cog-model/AmbiK-dataset. The calibration/test split in this repository is the one used in our experiments; it differs from the original AmbiK split. Each AmbiK row contains an ambiguous task and its unambiguous counterpart, so the AmbiK runners evaluate 1000 instructions from the 500 test rows. See the original repository for the dataset's license terms.
- **IntroPlan** — Liang et al., *Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity*, NeurIPS 2024. Original data: https://github.com/kevinliang888/IntroPlan. Distributed under the MIT License; see `data/LICENSE-IntroPlan`.

## License

No license has been chosen for the code yet. Until a license is added, the code is not licensed for reuse, modification or redistribution. The IntroPlan-derived data files are covered by `data/LICENSE-IntroPlan`.

## Citation

Citation information will be added after publication.
