
"""Gist compression demo."""
from typing import Optional
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaTokenizer


if Path('../../gisting').exists() or Path('../../../gisting').exists():
    import sys
    if Path('../../gisting').exists() and not '../..' in sys.path:
        sys.path.append('../..')
    if Path('../../../gisting').exists() and not '../../..' in sys.path: # for loading from src/notebooks
        sys.path.append('../../..')
    from gisting.src import gist_llama, gist_t5, weight_diff
    from gisting.src.gist_llama import GistLlamaForCausalLM
    from gisting.src.gist_t5 import GistT5ForConditionalGeneration
    from gisting.src.data.collator import get_prompt
else:
    print("Gist repository not found. Experiments involving gisting will fail.")
from constants import Dataset as D

get_is_t5 = lambda model_name_or_path: "t5" in model_name_or_path.lower() or True
get_is_llama = lambda model_name_or_path: "llama" in model_name_or_path.lower()

def get_model(
    model_name_or_path: str,
    cache_dir: str = ".cache",
    precision: str = "fp32",
    base_llama_path: Optional[str] = None,
    device: str = 'cuda:0',
):
    # Load config
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    is_llama = get_is_llama(model_name_or_path)
    is_t5 = get_is_t5(model_name_or_path)

    print(f"Loading model {model_name_or_path}")
    if is_t5:
        model_cls = GistT5ForConditionalGeneration
    elif is_llama:
        model_cls = GistLlamaForCausalLM
    else:
        raise ValueError(f"Model type {model_name_or_path} not supported")

    if model_name_or_path in {
        "jayelm/llama-7b-gist-1",
        "jayelm/llama-7b-pos_control-1",
        "jayelm/llama-7b-neg_control-1",
    }:
        # Load with weight diff file
        if base_llama_path is None:
            raise ValueError(
                f"{model_name_or_path} is a weight diff huggingface repo. "
                "You must specify a `base_llama_path` for this to work."
            )
        else:
            print("Weight diff detected. Applying to original model...")
        model, _ = weight_diff.recover(
            path_raw=base_llama_path,
            path_diff=model_name_or_path,
            test_inference=False,
            cache_dir=cache_dir,
        )
    else:
        model = model_cls.from_pretrained(
            model_name_or_path,
            config=config,
            cache_dir=cache_dir,
        )

    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float,
    }
    print(f"Converting model to {precision}")
    model = model.to(dtypes[precision]).to(device).eval()
    return model

def get_tokenizer(model_name_or_path: str, model):
    is_llama = get_is_llama(model_name_or_path)
    is_t5 = get_is_t5(model_name_or_path)

    print("Loading tokenizer")
    if is_llama:
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        assert len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 1
        assert model.lm_head.weight.shape[0] == gist_llama.PRETRAINED_VOCAB_SIZE + 1
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        assert len(tokenizer) == gist_t5.PRETRAINED_VOCAB_SIZE + 1
        assert model.shared.weight.shape[0] == gist_t5.PRETRAINED_VOCAB_SIZE + 1
    gist_token = tokenizer.additional_special_tokens_ids[-1]
    return tokenizer, gist_token


class Gister:
    def __init__(
        self,
        dataset: D,
        model_name_or_path,
        num_gist_tokens: int,
        cache_dir: str = ".cache",
        precision: str = "fp32",
        device: str = 'cuda:0',
        base_llama_path: Optional[str] = None,
    ):
        self.dataset = dataset
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.num_gist_tokens = num_gist_tokens
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, use_fast=True,
        )
        self.model = self.get_model(
            precision=precision,
            base_llama_path=base_llama_path,
        )
        self.tokenizer, self.gist_token = self.get_tokenizer()

    @property
    def is_llama(self):
        return get_is_llama(self.model_name_or_path)

    @property
    def is_t5(self):
        return get_is_t5(self.model_name_or_path)

    def get_model(self, precision, base_llama_path):
        return get_model(self.model_name_or_path, self.cache_dir, precision, base_llama_path, self.device)

    def get_tokenizer(self):
        return get_tokenizer(self.model_name_or_path, self.model)

    def compress(self, prompt):
        instruction_input_ids = self.tokenizer.encode(prompt)

        if self.is_t5:
            instruction_input_ids = instruction_input_ids[:-1]  # Remove eos token
        instruction_input_ids_tensor = (
            torch.tensor(instruction_input_ids).unsqueeze(0).to(self.device)
        )
        gist_kwargs = {
            "input_ids": instruction_input_ids_tensor,
            "attention_mask": torch.ones_like(instruction_input_ids_tensor),
        }
        if self.is_llama:
            gist_kwargs["attention_mask_gist"] = torch.ones_like(
                instruction_input_ids_tensor
            )[None, None]
        gist_activations = self.model.get_gist_activations(
            gist_token=self.gist_token,
            num_gist_tokens=self.num_gist_tokens,
            **gist_kwargs,
        )
        return gist_activations


def compress_using_model(
    dataset: D, instance, gist_token, num_gist_tokens,
    model, tokenizer, is_t5, is_llama
):
    gist_str = "<GIST>" * num_gist_tokens
    prepped_instruction = get_prompt(dataset, instance, gist_str)
    instruction_input_ids = tokenizer.encode(prepped_instruction)

    if is_t5:
        instruction_input_ids = instruction_input_ids[:-1]  # Remove eos token
    instruction_input_ids_tensor = (
        torch.tensor(instruction_input_ids).unsqueeze(0).cuda()
    )
    gist_kwargs = {
        "input_ids": instruction_input_ids_tensor,
        "attention_mask": torch.ones_like(instruction_input_ids_tensor),
    }
    if is_llama:
        gist_kwargs["attention_mask_gist"] = torch.ones_like(
            instruction_input_ids_tensor
        )[None, None]
    gist_activations = model.get_gist_activations(
        gist_token=gist_token,
        num_gist_tokens=num_gist_tokens,
        **gist_kwargs,
    )
    return gist_str, instruction_input_ids, gist_activations