import gc
import numpy as np

import parser as response_parser
import metrics


class KnowNoPipe:
    """KnowNo pipeline: generate 4 options, then a second LLM call scores
    each option's first-token logit, and self.cp thresholds which options
    survive (conformal prediction) — the actual decision logic
    (answer_with_cp) is untouched from the original.

    Args:
        model: an already-loaded llm.LLM instance (e.g. from
            load_llm(MODEL_NAME)) — loaded ONCE by the caller and reused
            for both the generation and answering calls. The original
            supported two separate models (examples_generation vs
            answering) via a two-role config; simplified to one model
            here, matching LoFreeCP and matching how the model was
            actually being run in practice (both roles pointed at the
            same checkpoint).
        cp: conformal prediction threshold. Defaults to the original
            hardcoded 0.794 if not given. Pass the value loaded from
            calibration_params_<dataset>.csv to use a calibrated
            threshold instead.
    """
    def __init__(self, model, cp=None):
        self.model = model
        self.cp = cp if cp is not None else 0.794
        self.mapping_1 = ['A', 'B', 'C', 'D']

    def options_prompt(self, description, task, prefix, action):
        return response_parser.load_prompt(
            "knowno", self.model.model_name, "generation",
            description=description, task=task, prefix=prefix, act=action,
        )

    def format_options(self, raw_text):
        """Was: format_examples() — a local regex parser identical in
        shape to what's now parser.py's parse_options_as_dict (same
        letter/digit matching, same "do nothing" default for missing
        letters). Consolidated there instead of duplicated here."""
        return response_parser.parse_options_as_dict(raw_text)

    def predict_examples(self, description, task, prefix, action):
        prompt = self.options_prompt(description, task, prefix, action)
        raw_text = self.model.generate(prompt)
        return self.format_options(raw_text)

    def answer_with_cp(self, tokens_logits):
        """Unchanged conformal-prediction decision: which option letters'
        first-token logit clears self.cp. NOTE (pre-existing, not
        introduced here): tokens_logits are softmax-normalized over 4
        candidates (filter_logits(..., use_softmax=True)), so they sum to
        ~1 — if self.cp sits above 0.5, at most one letter can ever clear
        it, making help_rate > 0 mathematically impossible regardless of
        the actual sample. The hardcoded default (0.794) has exactly this
        problem. This isn't fixed here (same "don't touch the algorithm"
        boundary as LoFreeCP) — but once real calibration is wired in
        (below), the calibrated qhat comes from the same softmax-normalized
        distribution this threshold is compared against, so it should
        land somewhere sane on its own. Sanity-check the calibrated value
        before trusting it either way.
        """
        possible_options = []
        for key in tokens_logits.keys():
            if tokens_logits[key] > self.cp:
                possible_options.append(key)

        formated_options = []
        for option in possible_options:
            if option.isdigit():
                option_formated = self.mapping_1[int(option) - 1]
            else:
                option_formated = option.upper()
            if option_formated not in formated_options:
                formated_options.append(option_formated)
        return formated_options

    def answer_prompt(self, options_dict, description, task, prefix, action):
        opts = ''
        for key, value in options_dict.items():
            opts += key + ') ' + value[3:] + '\n'
        return response_parser.load_prompt(
            "knowno", self.model.model_name, "choising",
            description=description, task=task, prefix=prefix, act=action,
            options=opts,
        )

    def generate_answer(self, prompt):
        text, logits = self.model.generate(prompt, return_logits=True)
        filtered_logits = self.model.filter_logits(
            logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4']
        )
        return filtered_logits, self.answer_with_cp(filtered_logits)

    def run(self, description, task, prefix, action):
        """Returns (option_dict, selected_option_texts) — selected_option_texts
        is already the CP-filtered result (answer_with_cp's output mapped
        back to option text), same as the original. Unlike LoFreeCP's
        pipeline.py (which discarded this and had to be fixed), KnowNo's
        run() already threaded self.cp's output through correctly — see
        calculate_metrics() below for exactly which of the two return
        values each metric uses."""
        mapdi = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}

        option = self.predict_examples(description, task, prefix, action)
        answer_prompt = self.answer_prompt(option, description, task, prefix, action)
        logits, answers = self.generate_answer(answer_prompt)

        answers_ = []
        for letter in answers:
            if letter in option.keys():
                answers_.append(option[letter])
            elif mapdi.get(letter, '') in option.keys():
                answers_.append(option[mapdi[letter]])
        return option, answers_

    def calculate_metrics(self, option, selected_answers, amb_type, user_intent, help_expected_types, aligned_fn):
        """SR reads `option` (the full 4-option pool, generated before CP
        filtering) — deliberately NOT gated by self.cp, mirroring
        LoFreeCP's SR (candidate-generation quality, independent of the
        help decision). help_rate/correct_help_rate/IA read
        `selected_answers` (the CP-filtered subset from run()) — these
        genuinely depend on self.cp.
        """
        full_pool = list(option.values())
        sr = metrics.success_rate(full_pool, user_intent)

        hr = metrics.hr_from_options(selected_answers)
        chr_value = metrics.correct_help_rate_from_options(selected_answers, amb_type, help_expected_types)
        ia = metrics.intent_alignment(hr, selected_answers, user_intent, aligned_fn)

        return {
            'SR': sr,
            'help_rate': float(hr),
            'correct_help_rate': float(chr_value),
            'IA': ia,
        }