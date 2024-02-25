
This package is to provide a codebase to easily fine-tune some open-source LLMs and reproduce the results. 

So far, the codebase supports 

- Fine-tuning algorithms offered by the [PEFT](https://github.com/huggingface/peft) package
  - LoRA
- Open-source LLM families from Hugging Face
  - [Llama2](https://huggingface.co/docs/transformers/main/en/model_doc/llama2)
  - [Mistral](https://huggingface.co/docs/transformers/main/en/model_doc/mistral)
  - [Falcon](https://huggingface.co/docs/transformers/main/en/model_doc/falcon)
  - [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5)
  - [Gemma](https://huggingface.co/docs/transformers/main/en/model_doc/gemma)
- In-context learning

For detailed instruction, please refer to the Notion site: [LLM Fine-tuning](https://yangfengji.notion.site/UVA-LLM-Fine-tuning-b5a80d6401e24ec6bb7900c4a3400918?pvs=4)

---

## Environment Setup

We provide the instructions for setting up the environment for 

- [UVA Rivanna](docs/rivanna.md), and 
- [UVA CS Servers](docs/cs-servers.md)

---

## A Simple Example

Here is a simple example of using the package for fine-tuning or prediction on the SST2 dataset. You can also find this demo code in `demo.py`. 

```python
import fire

from lora import Llama_Lora


def main(
        task: str = "eval",
		base_model: str = "meta-llama/Llama-2-7b-hf",
	):
    m = Llama_Lora(
        base_model=base_model,
    )
    if task == "train":
        m.train(
            train_file = "data/sst2/train.json",
            val_file = "data/sst2/val.json",
            output_dir = "./ckp_sst_llama2_lora",
            train_batch_size = 32,
            num_epochs = 1,
        )
    elif task == "eval":
        m.predict(
            input_file = "data/sst2/val.json",
            max_new_tokens = 32,
            verbose = True,
        )
    else:
        raise ValueError(f"Unrecognized task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
```

For prediction, please use 
```bash
python demo.py
```

For fine-tuning, please use 
```bash
python demo.py --task train
```

---

## Contributors

- [Yangfeng Ji](https://yangfengji.net)
- [Wanyu Du](https://wyu-du.github.io)
- [Aidan San](https://aidansan.github.io)
- [Zhengguang Wang](https://zhengguangw.github.io)


In addition, this package has been tested by many members from the [UVA ILP](https://uvanlp.org) group via their research projects.
