# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import LlamaForCausalLM, LlamaTokenizer
from trl import create_reference_model
from dpo_trainer import DPOTrainer

from accelerate import Accelerator, dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.01, metadata={"help": "the beta parameter for DPO loss"})
    
    # aps parameters
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": ""})
    rho: Optional[float] = field(default=0.1, metadata={"help": ""})
    tau_max: Optional[float] = field(default=5.0, metadata={"help": ""})
    tau_min: Optional[float] = field(default=0.1, metadata={"help": ""})

    # training parameters
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})

    eval_steps: Optional[int] = field(default=2800, metadata={"help": "eval steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "num epoch"})

    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    # lora parameters
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    train_path: Optional[str] = field(
        default=None,
    )
    val_path: Optional[str] = field(
        default=None,
    )
    output_path: Optional[str] = field(
        default=None,
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]

def construct_summarize_prompt(prompt):
    subreddit = prompt['subreddit'] if prompt['subreddit'] else ''
    title = prompt['title'] if prompt['title'] else ''
    post = prompt['post'] if prompt['post'] else ''
    # return "SUBREDDIT: " + prompt['subreddit'] + " " + "TITLE: " + prompt['title'] + " " + "POST: " + prompt['post']
    return "SUBREDDIT: " + subreddit + " " + "TITLE: " + title + " " + "POST: " + post

def get_summarize(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = '~/.cache/huggingface/datasets') -> Dataset:
    print(f'Loading OpenAI summarization dataset ({split} split) from Huggingface...')
    dataset = load_dataset('openai/summarize_from_feedback', 'comparisons', split=split, cache_dir=cache_dir)
    print('done')
    # # select only the first 1000 samples
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": construct_summarize_prompt(sample['info']),
            "chosen": sample["summaries"][sample["choice"]]['text'],
            "rejected": sample["summaries"][1 - sample["choice"]]['text'],
        }

    return dataset.map(split_prompt_and_responses)


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return Dataset.from_dict(data)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path,
                                                low_cpu_mem_usage=True,
                                                torch_dtype=torch.bfloat16,
                                                )

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    model_ref = create_reference_model(model)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    # 2. Load the dataset
    train_dataset = load_json(script_args.train_path)
    # 3. Load evaluation dataset
    eval_dataset = load_json(script_args.val_path)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_train_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10,  # match results in blog post # is 10 in last submit
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_path,
        optim="rmsprop",
        warmup_steps=150,
        report_to=script_args.report_to,
        bf16=True,
        save_total_limit=1,
        gradient_checkpointing=script_args.gradient_checkpointing,
        save_strategy="epoch"
        # TODO: uncomment that on the next transformers release
        # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
    )

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    if script_args.loss_type == "aps":
        print("Using aps !!!!!")
    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=False,
        peft_config=peft_config,
        loss_type=script_args.loss_type,
        rho=script_args.rho,
        tau_max=script_args.tau_max,
        tau_min=script_args.tau_min,
        # ddp_find_unused_parameters=False
        # precompute_ref_log_probs = True
    )

    # 6. train
    dpo_trainer.train()









