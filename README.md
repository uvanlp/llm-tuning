
This package is to provide a codebase to easily fine-tune some open-source LLMs and reproduce the results. 

So far, the codebase supports 

- Fine-tuning algorithms offered by the [PEFT](https://github.com/huggingface/peft) package
  - LoRA
- Open-source LLM families from Hugging Face
  - Llama 2
  - Falcon
  - Flan-T5
- In-context learning

---

## Environment Setup

We provide the instructions for setting up the environment for 

- [UVA Rivanna](https://www.notion.so/Environment-Configuration-on-Rivanna-5cb1f289049146e6ae63546031df6498?pvs=4), and 
- [UVA CS Servers]()

---

## A Simple Example

Here is a simple example of using the package for fine-tuning or prediction on the SST2 dataset. You can also find this demo code in `demo.py`. 

```python
import fire

from lora import Llama_Lora


def main(
        task: str = "eval",
):
    base_model_name: str = "meta-llama/Llama-2-7b-hf"
    m = Llama_Lora(
        base_model=base_model_name,
    )
    if task == "train":
        m.train(
            train_file = "data/sst2/train.json",
            val_file = "data/sst2/val.json",
            output_dir = "./ckp_sst_llama2_lora",
            train_batch_size = 32,
            num_epochs = 1,
            group_by_length = False,
            logging_steps = 5,
            val_steps = 20,
            val_batch_size = 8,
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


In addition, this package has been tested by many members from the [UVA ILP] group via their research projects.
