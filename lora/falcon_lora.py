"""
falcon_lora.py

The class for fine-tuning Falcon with LoRA
"""
import os
import torch, transformers
from typing import List

# The corresponding Transformer module
from transformers import AutoModelForCausalLM, AutoTokenizer

# The LLM_Lora base class
from .llm_lora import LLM_Lora


class Falcon_Lora(LLM_Lora):
    def __init__(self,
                 base_model: str = "",
                 lora_target_modules: List[str] = ["query_key_value","dense"],
                 load_in_8bit: bool = True,
                 cutoff_length: int = 256
                 ):
        LLM_Lora.__init__(self,
                          base_model = base_model,
                          lora_target_modules = lora_target_modules,
                          load_in_8bit = load_in_8bit,
                          cutoff_length = cutoff_length,
                          )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    def load_base_model(self):
        if len(self.base_model) == 0:
            raise ValueError(f"Need to specify a Falcon pre-trained model -- the current base model is {self.base_model}")
        print(f"Load the pre-trained model: {self.base_model}")
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        
