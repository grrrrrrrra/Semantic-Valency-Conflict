import gc
import math
import numpy as np
import pandas as pd
import os
import re
from scipy import spatial
from gensim.models import FastText
from gensim.test.utils import common_texts

import parser as response_parser
import metrics

class LoFreePipe:
    """Main pipeline class for generating and evaluating task options with conformal prediction."""
    def __init__(self, model, cp=None):
        """Initialize the pipeline.

        Args:
            model: an already-loaded llm.LLM instance (e.g. from
                load_llm(MODEL_NAME)) — loaded ONCE by the caller and
                reused across every sample. Previously generate_options()
                constructed a brand new LLM(...) — reloading the entire
                model from disk — on every single call; that's gone now.
            cp: conformal prediction threshold. Defaults to the same
                hardcoded value as before (0.8970238474379625) if not
                given. Pass the value loaded from
                calibration_params_<dataset>.csv to actually use a
                calibrated threshold — previously calibrate.py computed
                and saved this but nothing ever read it back in, so the
                hardcoded default was always used regardless of
                calibration.
        """
        self.model = model
        self.prompting_number = 4  # Number of times to prompt LLM for options
        self.lambda1 = 0.1  # Weight for entropy in nonconformity score
        self.lambda2 = 0.1  # Weight for similarity in nonconformity score
        self.cp = cp if cp is not None else 0.8970238474379625  # Conformal prediction threshold
        # Initialize FastText model for semantic similarity
        self.model_ss = FastText(sentences=common_texts, vector_size=200, window=5, min_count=1, workers=4)
        self.mapping = {'A': '1', 'B': '2', 'C': '3', 'D': '4'}  # Option mapping

    def run(self, sample):
        """Execute the full pipeline for a given sample.

        Args:
            sample: Dictionary containing sample data

        Returns:
            tuple: (processed sample, list of selected options)
        """
        sample = self.generate_options(sample)  # sample['options'] = full generated pool, all 4 rounds, unfiltered
        sample = self.frequency(sample)
        normalized_entropy = self.normalized_entropy(sample['frequences'])
        sample = self.semantic_similarity(sample)
        sample = self.nonconformity_score(sample, normalized_entropy)
        sample, selected_options = self.answer_with_cp(sample)
        sample['selected_options'] = selected_options
        return sample, selected_options

    def generate_options(self, sample):
        """Generate multiple options for completing the task using LLM.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            Sample dictionary with generated options
        """
        prompt = self.build_prompt(sample)
        answers = {}

        for _ in range(self.prompting_number):
            text = self.model.generate(prompt)
            options = self._parse_response(text)
            for option in options:
                option_clean = option.lower().strip()
                answers[option_clean] = answers.get(option_clean, 0) + 1

        sample['options'] = list(answers.keys())
        sample['answers'] = answers
        return sample

    def build_prompt(self, sample):
        """Construct the prompt for LLM based on sample data.

        Args:
            sample: Dictionary containing sample data

        Returns:
            Formatted prompt string
        """
        prefix = sample.get('prefix', '')
        act = sample.get('act', sample['task'])
        constraints = (
            "Consider possible constraints before choosing."
            if sample.get('ambiguity_type') in ['spatial_ambiguous_task', 'unsafe_task']
            else ""
        )

        return response_parser.load_prompt(
            "lofreecp", self.model.model_name, "generation",
            description=sample['description'], task=sample['task'],
            prefix=prefix, act=act, constraints=constraints,
        )

    def _parse_response(self, text):
        """Parse LLM response to extract options.

        Args:
            text: Raw LLM response text

        Returns:
            List of unique parsed options
        """
        
        return response_parser.parse_options_as_list(text)

    def frequency(self, sample):
        """Calculate frequency distribution of generated options.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            Sample dictionary with frequency distribution
        """
        total = sum(sample['answers'].values())
        sample['frequences'] = {k: v/total for k, v in sample['answers'].items()}
        return sample

    def normalized_entropy(self, frequences):
        """Calculate normalized entropy of option frequencies.
        
        Args:
            frequences: Dictionary of option frequencies
            
        Returns:
            Normalized entropy value (float)
        """
        if not frequences:
            return 0.0
        entropy = -sum(p * math.log(p) for p in frequences.values())
        max_entropy = math.log(len(frequences))
        return abs(entropy / max_entropy) if max_entropy > 0 else 0.0

    def semantic_similarity(self, sample):
        """Calculate semantic similarity between options.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            Sample dictionary with similarity scores
        """
        vectors = {}
        for option in sample['options']:
            words = option.split()
            vectors[option] = sum(self.model_ss.wv[word] for word in words if word in self.model_ss.wv)
            if vectors[option].any():
                vectors[option] /= len(words)

        if not sample['frequences']:
            sample['similarities'] = {}
            return sample

        top_option = max(sample['frequences'], key=sample['frequences'].get)
        similarities = {}
        for option, vector in vectors.items():
            if option == top_option:
                similarities[option] = 0.0
            else:
                similarities[option] = 1 - spatial.distance.cosine(vector, vectors[top_option])

        sample['similarities'] = similarities
        return sample

    def nonconformity_score(self, sample, normalized_entropy):
        """Calculate nonconformity scores for options.
        
        Args:
            sample: Dictionary containing sample data
            normalized_entropy: Normalized entropy value
            
        Returns:
            Sample dictionary with nonconformity scores
        """
        scores = {}
        for option in sample['options']:
            freq_score = 1 - sample['frequences'][option]
            sim_score = self.lambda2 * sample['similarities'][option]
            ent_score = self.lambda1 * normalized_entropy
            scores[option] = freq_score + ent_score - sim_score

        sample['nonconformity_scores'] = scores
        return sample

    def answer_with_cp(self, sample):
        """Select options using conformal prediction threshold.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            tuple: (sample, list of selected options)
        """
        options = []
        scores = sample['nonconformity_scores']
        for option, score in scores.items():
            if score >= self.cp:
                options.append(option)
        return sample, options if options else []

    def calculate_metrics(self, sample, amb_type, user_intent, help_expected_types, aligned_fn):
        """Calculate metrics for a single sample.

        Args:
            sample: Processed sample dictionary
            amb_type: Ambiguity type of the task
            user_intent: List of user intents
            help_expected_types: which ambiguity_type values expect help
                for the dataset this sample is from (e.g.
                metrics.AMBIK_HELP_EXPECTED / metrics.INTROPLAN_HELP_EXPECTED)
                — added because this pipeline is now shared across both
                datasets, so it can no longer decide this on its own.
            aligned_fn: cosine-similarity alignment check for IA (e.g.
                is_intent_aligned_cosine bound to an embedding model) —
                same mechanism SVC uses, passed in because the embedding
                model lives at the runner level, not inside this class.

        Returns:
            Dictionary of calculated metrics
        """
        return {
            'SR': self.success_rate(sample, user_intent),
            'help_rate': self.help_rate(sample),
            'correct_help_rate': self.correct_help_rate(sample, amb_type, help_expected_types),
            'IA': self.intent_alignment(sample, user_intent, aligned_fn),
        }

    def success_rate(self, sample, user_intent):
        """Calculate success rate (matching user intents).

        Deliberately reads sample['options'] (the FULL generated pool),
        not sample['selected_options'] — SR is meant to measure
        candidate-generation quality independent of the CP decision (see
        help_rate/correct_help_rate/intent_alignment below, which DO move
        with self.cp). If SR were gated by CP too, a confident-and-wrong
        single-option answer (hr==0) would look identical to a
        confident-and-right one — exactly the false-negative signal SR
        exists to preserve.

        Args:
            sample: Processed sample dictionary
            user_intent: List of user intents

        Returns:
            Success rate (float)
        """
        return metrics.success_rate(sample['options'], user_intent)

    def help_rate(self, sample):
        """Calculate help rate (multiple options provided).

        Reads sample['selected_options'] — the options that actually
        survived the self.cp conformal-prediction threshold (run()
        stores this). Was reading sample['options'] (the full unfiltered
        pool) instead, which meant self.cp had literally zero effect on
        this metric's value regardless of what it was set to — calibrate
        a threshold, hardcode one, doesn't matter, same number either
        way. Falls back to sample['options'] only if run() wasn't used to
        produce this sample (shouldn't happen via the normal runner
        flow, but avoids a hard KeyError if called some other way).

        Args:
            sample: Processed sample dictionary

        Returns:
            Help rate (0.0 or 1.0)
        """
        options = sample.get('selected_options', sample['options'])
        return float(metrics.hr_from_options(options))

    def correct_help_rate(self, sample, amb_type, help_expected_types):
        """Calculate correct help rate (appropriate for ambiguity type).

        Reads sample['selected_options'] — same reasoning as help_rate()
        above: this now genuinely depends on self.cp.

        Args:
            sample: Processed sample dictionary
            amb_type: Ambiguity type of the task
            help_expected_types: which ambiguity_type values expect help
                for this dataset — see calculate_metrics().

        Returns:
            Correct help rate (0.0 or 1.0)
        """
        options = sample.get('selected_options', sample['options'])
        return float(metrics.correct_help_rate_from_options(options, amb_type, help_expected_types))

    def intent_alignment(self, sample, user_intent, aligned_fn):
        """IA — same function and same mechanism as SVC's IA
        (metrics.intent_alignment: NaN unless help was actually offered,
        checked via a pluggable aligned_fn). `aligned_fn` is real cosine
        similarity here too (the runner binds it to an embedding model),
        not a substring-matching stand-in — SVC and LoFreeCP now compute
        IA identically, just over different candidate sets (SVC's frame
        variants vs LoFreeCP's generated options).

        Reads sample['selected_options'] — same reasoning as help_rate()
        above: hr and the candidates checked for alignment both now
        genuinely depend on self.cp, not the full unfiltered pool.

        Args:
            sample: Processed sample dictionary
            user_intent: List of user intents
            aligned_fn: cosine-similarity alignment check, e.g.
                lambda intents, candidates: is_intent_aligned_cosine(
                    intents, candidates, embedding_model)

        Returns:
            IA (1.0, 0.0, or NaN if not applicable)
        """
        options = sample.get('selected_options', sample['options'])
        hr = metrics.hr_from_options(options)
        return metrics.intent_alignment(hr, options, user_intent, aligned_fn)