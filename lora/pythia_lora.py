"""
pythia_lora.py

The class for fine-tuning Pythia with LoRA
"""

import torch, transformers
from typing import List

from transformers import GPTNeoXForCausalLM, AutoTokenizer

# The LLM_Lora base class
from .llm_lora import LLM_Lora

import os
os.environ["TOKENIZER_PARALLELISM"] = "false"

# For some reason pythia should not be loaded in 8bit
# RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float

class Pythia_Lora(LLM_Lora):
    def __init__(self,
                 base_model: str = "",
                 lora_target_modules: List[str] = ["query_key_value", "dense"],
                 load_in_8bit: bool = False,
                 cutoff_length: int = 256,
                 ):
        LLM_Lora.__init__(self,
                          base_model = base_model,
                          lora_target_modules = lora_target_modules,
                          load_in_8bit = load_in_8bit,
                          cutoff_length = cutoff_length,
                          )


    def load_base_model(self):
        # Load the model for preparation
        if len(self.base_model) == 0:
            raise ValueError(f"The base_model is {self.base_model}")
        print(f"Loading the pre-trained model: {self.base_model}")

        self.model = GPTNeoXForCausalLM.from_pretrained(
            self.base_model,
            revision="step143000",
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # You can print the model to determine the lora_target_modules
        # print(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
