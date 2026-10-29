import gc
import numpy as np

import parser as response_parser
import metrics


class AffordancesPipe:
    """LAP/Affordances pipeline: generate 4 options, score each via a
    choosing-style logit call AND an independent "is this possible and
    safe" affordance check (LLM-probability x rule-based context check x
    choosing-logit), then self.cp thresholds the combined affordance_score
    (conformal prediction) — same decision shape as KnowNo, plus the
    extra affordance-scoring step.

    Args:
        model: an already-loaded llm.LLM instance — loaded ONCE by the
            caller, reused for the generation/answering/affordance calls.
            The original supported three separate models (examples_
            generation / answering / affordance_generation via a
            three-role config, with shared-weights reuse when two roles
            named the same checkpoint); simplified to one model here,
            matching LoFreeCP/KnowNo and matching how these were actually
            being run in practice (same checkpoint for all three roles).
        environment_objects: list of known object names used by
            context_based_affordance() to sanity-check whether an
            option's mentioned object is actually in the scene. This is
            dataset-specific — AmbiK and IntroPlan have different object
            vocabularies (confirmed: zero overlap), so pass the matching
            list for whichever dataset you're running (environment_objects
            for AmbiK, environment_objects_introplan for IntroPlan — the
            latter built from introplan_test.csv/introplan_calib.csv's
            actual environment_short values, not guessed). Passing the
            wrong one means context_based_affordance() silently returns 1
            (its vacuous-truth default on an empty match) for every row,
            making the context-based component a no-op.
        cp: conformal prediction threshold. Defaults to the original
            hardcoded 0.7941411996916858 if not given. Pass the value
            loaded from calibration_params_<dataset>.csv to use a
            calibrated threshold instead.
    """
    def __init__(self, model, environment_objects, cp=None):
        self.model = model
        self.cp = cp if cp is not None else 0.7941411996916858
        self.mapping_1 = ['A', 'B', 'C', 'D']
        self.environment_objects = environment_objects

    def options_prompt(self, description, task, prefix, action):
        return response_parser.load_prompt(
            "lap", self.model.model_name, "generation",
            description=description, task=task, prefix=prefix, act=action,
        )

    def format_options(self, raw_text):
        """Was: format_examples() — a local regex parser, same shape as
        what's now parser.py's parse_options_as_dict. Consolidated there
        instead of duplicated here."""
        return response_parser.parse_options_as_dict(raw_text)

    def predict_examples(self, description, task, prefix, action):
        prompt = self.options_prompt(description, task, prefix, action)
        raw_text = self.model.generate(prompt)
        return self.format_options(raw_text)

    def context_based_affordance(self, objects, scene):
        """Unchanged: returns 0 if any of `objects` is missing from
        `scene`, else 1. NOTE: on an empty `objects` list (nothing in
        self.environment_objects matched the option text), this loop
        never runs and falls through to `return 1` — vacuously "fine".
        See the class docstring re: environment_objects' AmbiK-derived
        vocabulary possibly making this happen on every IntroPlan row."""
        for obj in objects:
            if obj not in scene:
                return 0
        return 1

    def prompt_based_affordance(self, description, action_text):
        """Returns the softmax-scaled PROBABILITY of whichever True/Yes/
        False/No word the model actually generated — not a boolean. This
        gets multiplied into affordance_score below, so it's meant to be
        a continuous signal, not a binary decision (parser.py's
        parse_affordance_response is for the simpler binary case
        elsewhere; this scoring role stays a bespoke method).

        `action_text` must be the SPECIFIC candidate option's text (e.g.
        "pick up the RedBull from the fridge"), not the overall sample
        task — this is what the affordances.txt prompt's <TASK> slot
        actually means (see its few-shot example: "open the bottle of
        honey without any tools" — a specific action, not a task
        description).

        FIXED (was flagged since the very first pass over this codebase):
        previously called with the overall sample description/task, so
        every one of the 4 options got the identical prompt_score
        regardless of what it actually proposed doing — the affordance
        check wasn't checking anything option-specific at all. Now called
        once per option with that option's own text (see
        generate_affordances() below), so "is THIS possible and safe?" is
        actually being asked about each candidate action individually.
        """
        prompt = response_parser.load_prompt(
            "lap", self.model.model_name, "affordances",
            description=description, task=action_text,
        )
        generated_text, logits = self.model.generate(prompt, return_logits=True)
        generated_text = generated_text.strip().split('\n')[0]

        allowed_words = ['True', 'Yes', 'False', 'No']
        if generated_text in allowed_words:
            filtered = self.model.filter_logits(logits[-1][0], words=[generated_text])
            return filtered.get(generated_text, 0.0)
        try:
            filtered = self.model.filter_logits(logits[-1][0], words=allowed_words)
            return max(filtered.values()) if filtered else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _strip_label(letter, value):
        prefix = f"{letter}) "
        if isinstance(value, str) and value.startswith(prefix):
            return value[len(prefix):]
        return value

    def generate_affordances(self, description, task, options, logits):
        """options: the dict from predict_examples() (letter -> "X) text",
        or the "do nothing" default for a missing letter). logits: the
        processed choosing-step logits (letter -> probability), from
        generate_answer() below. `task` is accepted for signature
        compatibility with the original but no longer used here — each
        option's own text is what actually gets checked now (see
        prompt_based_affordance()'s docstring for why).
        """
        li = ['A', 'B', 'C', 'D']
        context_scores, prompt_scores, affordance_scores = {}, {}, {}

        for letter in li:
            option_text = self._strip_label(letter, options.get(letter, 'do nothing'))

            prompt_score = float(self.prompt_based_affordance(description, option_text))

            cur_objects = [obj for obj in self.environment_objects if obj in option_text]
            context_score = float(self.context_based_affordance(cur_objects, description))

            context_scores[letter] = context_score
            prompt_scores[letter] = prompt_score

            if letter in logits:
                try:
                    logit_val = float(logits[letter])
                    affordance_score = prompt_score * context_score * (1 + logit_val)
                except Exception:
                    affordance_score = 0.0
            else:
                affordance_score = 0.0
            affordance_scores[letter] = affordance_score

        return context_scores, prompt_scores, affordance_scores

    def answer_with_cp(self, affordance_scores):
        """Unchanged conformal-prediction decision (>=, not > — matches
        this pipeline's own original convention, distinct from KnowNo's
        strict >)."""
        possible_options = []
        for key, value in affordance_scores.items():
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = 0.0
            if score >= self.cp:
                possible_options.append(key)

        formatted_options = []
        for option in possible_options:
            if option.isdigit() and 1 <= int(option) <= 4:
                option_formatted = self.mapping_1[int(option) - 1]
            elif option.upper() in ['A', 'B', 'C', 'D']:
                option_formatted = option.upper()
            else:
                continue
            if option_formatted not in formatted_options:
                formatted_options.append(option_formatted)
        return formatted_options

    def answer_prompt(self, options_dict, description, task, prefix, action):
        opts = ''
        for key, value in options_dict.items():
            opts += key + ') ' + self._strip_label(key, value) + '\n'
        return response_parser.load_prompt(
            "lap", self.model.model_name, "choising",
            description=description, task=task, prefix=prefix, act=action,
            options=opts,
        )

    def generate_answer(self, description, task, prefix, action, options):
        prompt = self.answer_prompt(options, description, task, prefix, action)
        text, logits = self.model.generate(prompt, return_logits=True)
        filtered_logits = self.model.filter_logits(
            logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4']
        )

        processed_logits = {}
        mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
        for key, value in filtered_logits.items():
            clean_key = str(key).upper()
            if clean_key in mapping:
                clean_key = mapping[clean_key]
            if clean_key in ['A', 'B', 'C', 'D']:
                processed_logits[clean_key] = value

        context_scores, prompt_scores, affordance_scores = self.generate_affordances(
            description, task, options, processed_logits
        )
        return processed_logits, affordance_scores, self.answer_with_cp(affordance_scores)

    def run(self, description, task, prefix, action):
        """Returns (option_dict, selected_option_texts) — selected_option_texts
        is the CP-filtered result (answer_with_cp's output mapped back to
        option text)."""
        options = self.predict_examples(description, task, prefix, action)
        gc.collect()
        logits, affordance_scores, cp_ans = self.generate_answer(description, task, prefix, action, options)

        selected_texts = []
        for letter in cp_ans:
            if letter in options:
                selected_texts.append(options[letter])
        return options, selected_texts

    def calculate_metrics(self, option, selected_answers, amb_type, user_intent, help_expected_types, aligned_fn):
        """SR reads `option` (the full 4-option pool, generated before CP
        filtering) — deliberately NOT gated by self.cp, mirroring
        LoFreeCP/KnowNo's SR. help_rate/correct_help_rate/IA read
        `selected_answers` (the CP-filtered subset from run()) — these
        genuinely depend on self.cp."""
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