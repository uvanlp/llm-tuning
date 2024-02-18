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
            # lora_adapter = "./ckp_sst2_llama2_lora",
            max_new_tokens = 32,
            verbose = True,
        )
    else:
        raise ValueError(f"Unknown model name: {base_model_name}")


if __name__ == "__main__":
    fire.Fire(main)
