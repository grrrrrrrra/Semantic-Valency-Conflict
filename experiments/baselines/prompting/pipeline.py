import parser as response_parser
import metrics


class PromptingPipe:
    """Prompting baseline: a single LLM call per instruction, asking for
    a JSON {"label": "Ambiguous"|"Unambiguous", "ambiguities": [...]}
    verdict directly. No conformal prediction, no calibration threshold —
    unlike LoFreeCP/KnowNo/LAP, this baseline doesn't threshold anything,
    so there's no self.cp and no calibrate_*.py for it.

    Args:
        model: an already-loaded llm.LLM instance (e.g. from
            load_llm(MODEL_NAME)).
    """
    def __init__(self, model):
        self.model = model

    def build_prompt(self, instruction, environment):
        environment_str = ", ".join(environment) if isinstance(environment, list) else str(environment)
        return response_parser.load_prompt(
            "prompting", self.model.model_name, "generation",
            instruction=instruction, environment=environment_str,
        )

    def get_llm_ambiguity_label(self, instruction, environment):
        prompt = self.build_prompt(instruction, environment)
        response_text = self.model.generate(prompt)
        return response_parser.parse_prompting_response(response_text)

    @staticmethod
    def _extract_all_candidates(llm_result):
        all_candidates = []
        for ambiguity in llm_result.get('ambiguities', []):
            candidates = ambiguity.get('candidates', [])
            if isinstance(candidates, list):
                all_candidates.extend(c for c in candidates if isinstance(c, str))
        return all_candidates

    @staticmethod
    def parse_user_intent(user_intent_raw):
        """Unchanged from the original — splits on |, comma, ;, or ' and ',
        in that priority order."""
        import pandas as pd
        if pd.isna(user_intent_raw) or user_intent_raw is None:
            return []

        intent_text = str(user_intent_raw).lower().strip()
        if not intent_text:
            return []
        intent_text = intent_text.strip('"').strip("'")

        if '|' in intent_text:
            separator = '|'
        elif ',' in intent_text:
            separator = ','
        elif ';' in intent_text:
            separator = ';'
        elif ' and ' in intent_text:
            separator = ' and '
        else:
            return [intent_text]

        return [item.strip() for item in intent_text.split(separator) if item.strip()]

    def calculate_metrics(self, instruction, environment, user_intent_raw, amb_type,
                           help_expected_types, aligned_fn):
        """Args:
            instruction, environment: the row's task text and object list
            user_intent_raw: raw user_intent cell (parsed internally)
            amb_type: this row's ambiguity_type
            help_expected_types: which ambiguity_type values expect help
                for the dataset this row is from (e.g.
                metrics.AMBIK_HELP_EXPECTED / metrics.INTROPLAN_HELP_EXPECTED)
                — was hardcoded to `ambiguity_type == 'preferences'`
                before, which is correct for AmbiK but silently wrong for
                every IntroPlan row (IntroPlan has no 'preferences'
                category at all, so this now-removed check always fell
                through to "no help expected", corrupting CHR for the
                whole dataset).
            aligned_fn: cosine-similarity alignment check (e.g.
                is_intent_aligned_cosine bound to an embedding model) —
                same mechanism SVC/LoFreeCP/KnowNo/LAP use for IA now.
                Was substring/exact-match before.

        Returns dict with HR, CHR, IA, SR plus the raw LLM label/
        candidates for debugging.
        """
        llm_result = self.get_llm_ambiguity_label(instruction, environment)

        hr = metrics.hr_from_label(llm_result.get('label', 'Unambiguous'))
        chr_value = metrics.correct_help_rate_from_label(hr, amb_type, help_expected_types)

        candidates = self._extract_all_candidates(llm_result)
        user_intents = self.parse_user_intent(user_intent_raw)

        ia = metrics.intent_alignment(hr, candidates, user_intents, aligned_fn)

        sr = metrics.success_rate(candidates, user_intents)

        return {
            'HR': hr,
            'CHR': chr_value,
            'IA': ia,
            'SR': sr,
            'llm_response_label': llm_result.get('label', 'Unknown'),
            'llm_ambiguities_count': len(llm_result.get('ambiguities', [])),
            'llm_candidates': candidates,
        }