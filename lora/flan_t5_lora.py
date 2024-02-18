"""
flan_t5_lora.py

The class for fine-tuning Flan-t5 with LoRA
"""
import os
import torch, transformers
from typing import List

# The corresponding Transformer module
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import TaskType

# The LLM_Lora base class
from .llm_lora import LLM_Lora


class Flan_T5_Lora(LLM_Lora):
    def __init__(self,
                 base_model: str = "",
                 lora_target_modules: List[str] = ["q","v"],
                 load_in_8bit: bool = True,
                 cutoff_length: int = 256,
                 task_type: str = TaskType.SEQ_2_SEQ_LM
                 ):
        LLM_Lora.__init__(self,
                          base_model = base_model,
                          lora_target_modules = lora_target_modules,
                          load_in_8bit = load_in_8bit,
                          cutoff_length = cutoff_length,
                          task_type = task_type
                        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        
    def load_base_model(self):
        if len(self.base_model) == 0:
            raise ValueError(f"Need to specify a Flan-T5 pre-trained model -- the current base model is {self.base_model}")
        print(f"Load the pre-trained model: {self.base_model}")
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.base_model)
