import os
import yaml
import numpy as np
from collections import Counter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GemmaForCausalLM,
)
import torch
import threading

from openai import OpenAI 


class TimeoutException(Exception):
    pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self):
        raise TimeoutException(self.error_message)

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.handle_timeout)
        self.timer.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.cancel()


def temperature_scaling(logits, temperature=1):
    logits = np.array(logits)
    logits = logits / temperature
    try:
        logits = logits - logits.max()
    except Exception:
        pass
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return [float(x) for x in smx]


def _call_with_retry(fn, max_attempts=3, timeout_seconds=40):
    """Shared retry/timeout wrapper for OpenRouter calls."""
    last_error = None
    for _ in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):
                return fn()
        except TimeoutException as e:
            last_error = e
            continue
    raise last_error if last_error else RuntimeError("API call failed with no captured error")


def detect_model_family(model_name):
    """Which prompt-family a model belongs to: 'qwen' | 'mistral' | 'gpt' |
    'claude'. Finer-grained than LLM.__init__'s own family branching below
    (which lumps gpt/claude together as mtype='openrouter_api', since
    loading is identical for both) — prompt SELECTION needs to tell them
    apart, since they're different files. Single source of truth so
    runner scripts don't each re-implement this detection."""
    name = model_name.lower()
    if 'qwen' in name:
        return 'qwen'
    if 'mistral' in name:
        return 'mistral'
    if name.startswith('anthropic/') or 'claude' in name:
        return 'claude'
    if name.startswith('openai/') or 'gpt' in name:
        return 'gpt'
    raise ValueError(f"detect_model_family: no prompt family known for {model_name!r}")


class LLM:
    def __init__(self, model_name, generation_settings):
        self.device = "cuda"
        self.mtype = None
        self.model_name = model_name
        self.chat_template_kwargs = {}
        self.generation_settings = generation_settings

        name = model_name.lower()
        
        if name.startswith('openai/') or name.startswith('anthropic/'):
            self.model = model_name
            self.mtype = 'openrouter_api'
            api_key = generation_settings.pop('api_key', None) or os.environ.get('OPENROUTER_API_KEY')
            # self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                     api_key=api_key,
                     timeout=120.0,
                     max_retries=5)

        elif "t5" in name or "bart" in name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.mtype = 'd'

        elif 'gemma' in name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = GemmaForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.mtype = 'ed'

        elif 'mistral' in name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.mtype = 'chat'

        elif 'qwen' in name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.mtype = 'chat'
            self.chat_template_kwargs = {'enable_thinking': False}

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.mtype = 'ed'

        if self.mtype != 'openrouter_api':
            if not any(k in name for k in ('mistral', 'gemma', 'qwen')):
                self.model.to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.mtype in ('ed', 'chat'):
                self.tokenizer.padding_side = 'left'

    def _format_prompt(self, prompt):
        if self.mtype == 'chat':
            already_templated = prompt.lstrip().startswith(('<|im_start|>', '<s>[INST]', '[INST]'))
            if already_templated:
                return prompt
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        return prompt

    def get_full_logits(self, logits, words):
        token_ids = []
        filtered_logits = []
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            token_ids.append(word_token_ids)

        for option_ids in token_ids:
            ids = [t.item() for t in torch.tensor(option_ids, device=self.device)]
            count_tokens = dict(Counter(ids).most_common())
            token_ids_target = [key for key in count_tokens.keys() if count_tokens[key] == 1]
            filtered_logits.append([logits[t].item() for t in token_ids_target])

        return dict(zip(words, filtered_logits))

    def filter_logits(self, logits, words, use_softmax=True):
        if self.mtype == 'openrouter_api':
            filtered_logits = []
            filtered_letters = []
            for key, value in logits.items():
                for word in words:
                    if key == word:
                        filtered_logits.append(value)
                        filtered_letters.append(key)
            if use_softmax:
                filtered_logits = temperature_scaling(filtered_logits)
            return dict(zip(filtered_letters, filtered_logits))

        token_ids = []
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

        token_ids = [t.item() for t in torch.tensor(token_ids, device=self.device)]
        count_tokens = dict(Counter(token_ids).most_common())
        token_ids_target = [key for key in count_tokens.keys() if count_tokens[key] == 1]
        filtered_logits = [logits[t].item() for t in token_ids_target]
        if use_softmax:
            filtered_logits = temperature_scaling(filtered_logits)
        return dict(zip(words, filtered_logits))

    def generate(self, prompt, return_logits=False):
        if self.mtype == 'openrouter_api':
            return self._generate_openrouter(prompt, return_logits)
        return self._generate_local(prompt, return_logits)

    def _generate_local(self, prompt, return_logits):
        text_input = self._format_prompt(prompt)
        inputs = self.tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        generated_ids = self.model.generate(
            input_ids, attention_mask=attention_mask, **self.generation_settings
        )
        new_tokens = generated_ids[:, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        if not return_logits:
            return generated_text

        with torch.no_grad():
            if self.mtype in ('ed', 'chat'):
                combined_ids = torch.cat([input_ids, new_tokens], dim=1)
                outputs = self.model(combined_ids)
                logits = outputs.logits[:, input_ids.size(1):]
            else:
                outputs = self.model(input_ids=input_ids, decoder_input_ids=generated_ids[:, :-1])
                logits = outputs.logits

            special_tokens = self.tokenizer.all_special_ids
            mask = torch.ones_like(new_tokens[0], dtype=torch.bool)
            for token_id in special_tokens:
                mask &= (new_tokens[0] != token_id)

            filtered_logits = logits[:, mask]
        return (generated_text, filtered_logits)

    def _generate_openrouter(self, prompt, return_logits):
        def _call():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                n=1,
                logprobs=return_logits,
                top_logprobs=20 if return_logits else None,
                **self.generation_settings,
            )
        response = _call_with_retry(_call)
        generated_text = response.choices[0].message.content.strip()

        if not return_logits:
            return generated_text
        choice_logprobs = getattr(response.choices[0], 'logprobs', None)
        if not choice_logprobs or not getattr(choice_logprobs, 'content', None):
            raise RuntimeError(
                f"return_logits=True requested for '{self.model_name}' via "
                f"OpenRouter, but no logprobs came back in the response. "
                f"Anthropic models never return logprobs (any route) — "
                f"don't use Claude for logit-based baselines (KnowNo/LAP). "
                f"For OpenAI models, OpenRouter doesn't guarantee logprobs "
                f"pass through even when requested — verify with a single "
                f"test call before a full run, or call OpenAI's API "
                f"directly for logit-dependent runs instead of routing "
                f"through OpenRouter."
            )

        filtered_logits = {}
        for logit in choice_logprobs.content[0].top_logprobs:
            filtered_logits[logit.token] = logit.logprob
        return (generated_text, [[filtered_logits]])

    def generate_batch(self, prompts, return_logits=False):
        if self.mtype == 'openrouter_api':
            results = [self.generate(p, return_logits=return_logits) for p in prompts]
            if return_logits:
                texts, logits_li = zip(*results)
                return (list(texts), list(logits_li))
            return results

        text_inputs = [self._format_prompt(p) for p in prompts]
        inputs = self.tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        generated_ids = self.model.generate(
            input_ids, attention_mask=attention_mask, **self.generation_settings
        )
        new_tokens = generated_ids[:, input_ids.shape[1]:]
        generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in new_tokens]

        if not return_logits:
            return generated_texts

        logits = self._compute_logits(input_ids, generated_ids)
        filtered_logits = self._filter_special_token_logits(generated_ids, logits)
        return (generated_texts, filtered_logits)

    def _compute_logits(self, input_ids, generated_ids):
        with torch.no_grad():
            if self.mtype in ('ed', 'chat'):
                combined_ids = torch.cat([input_ids, generated_ids[:, input_ids.shape[1]:]], dim=1)
                outputs = self.model(combined_ids)
                logits = outputs.logits[:, input_ids.size(1):]
            else:
                outputs = self.model(input_ids=input_ids, decoder_input_ids=generated_ids[:, :-1])
                logits = outputs.logits
        return logits

    def _filter_special_token_logits(self, generated_ids, logits):
        special_tokens = self.tokenizer.all_special_ids
        new_tokens = generated_ids[:, -logits.shape[1]:]
        batch_size, seq_length = new_tokens.shape
        mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=logits.device)
        for token_id in special_tokens:
            mask &= (new_tokens != token_id)
        return torch.stack([logit[mask[i]] for i, logit in enumerate(logits)])


_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LLM_CONFIG_PATH = os.path.join(_LLM_DIR, "llm_config.yaml")


def load_llm(name, config_path=None):
    """Look up `name` in the llm_config.yaml registry and return a ready
    LLM instance — e.g. load_llm("qwen3-8b").

    config_path defaults to llm_config.yaml sitting next to THIS file
    (utilities/llm_config.yaml), resolved via __file__ rather than the
    current working directory — so this works the same whether you run a
    runner from experiments/SVC/, experiments/knowno/, or anywhere else,
    without needing to `cd` into utilities/ first or pass the path
    explicitly every time."""
    if config_path is None:
        config_path = _DEFAULT_LLM_CONFIG_PATH

    with open(config_path, "r") as f:
        registry = yaml.safe_load(f)

    if name not in registry["models"]:
        raise KeyError(
            f"'{name}' not found in {config_path}. Available: "
            f"{list(registry['models'].keys())}"
        )

    entry = registry["models"][name]
    return LLM(entry["model_name"], dict(entry.get("generation_settings", {})))


if __name__ == "__main__":
    model = load_llm("qwen3-8b")
    print(model.generate("Choose one letter A/B/C?"))