from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
import torch
from model import CustomModel
from datasets import load_dataset
from evaluation import eval_model
import os
from datetime import datetime


model_name = "microsoft/swinv2-tiny-patch4-window8-256"
image_processor = AutoImageProcessor.from_pretrained(model_name)


def transform(batch):
    inputs = image_processor(
        [x.convert("RGB") for x in batch["image"]], return_tensors="pt"
    )
    inputs["labels"] = batch["labels"]

    return inputs


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor(
            [x["labels"] for x in batch],
            dtype=torch.float,
        ),
    }


def main():
    model = CustomModel()

    ### CHANGE ME ###
    on_server = True

    is_testing = False
    num_samples = 19659  # 65526 / 4 -- if is_testing False, then this is irrelevant
    can_skip = True

    label_type = "chexpert"
    # label_type = "negbio"

    views = ["PA"]  # PA is front, lateral is side

    train = (
        False  # for just eval on the model at input_model_path, change this to False
    )
    input_model_path = None

    num_epochs = 12
    batch_size = 32
    learning_rate = 2e-4

    path_to_cache = ""
    path_to_data = ""
    #################

    if on_server:  # set cache info
        os.environ["HF_HOME"] = path_to_cache

    data_dir = path_to_data if on_server else "."

    os.makedirs("evaluation", exist_ok=True)

    experiment_dir = os.path.join(
        "evaluation", datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    )

    if train:
        output_dir = os.path.join(experiment_dir, "outputs")
        model_path = os.path.join(output_dir, "model_state_dict.pth")

        # we use a local dataset builder script (builder.py) -- it is cached in ./cache/huggingface after runs
        # therefore, if you make any changes to builder.py, close the terminal and run main.py in a fresh terminal
        train_dataset = load_dataset(
            "builder.py",
            data_dir=data_dir,
            label_type=label_type,
            views=views,
            is_testing=is_testing,
            can_skip=can_skip,
            on_server=on_server,
            num_samples=num_samples,
            split="train",
            trust_remote_code=True,
        ).with_transform(transform)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            fp16=torch.cuda.is_available(),
            save_steps=100,
            logging_steps=10,
            learning_rate=learning_rate,
            save_total_limit=2,
            remove_unused_columns=False,  # keep this to properly collate
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=train_dataset,
        )

        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        torch.save(model.state_dict(), model_path)

    else:
        model_path = input_model_path

    eval_model(
        model_path,
        experiment_dir,
        data_dir,
        views,
        label_type,
        is_testing,
        can_skip,
        on_server,
        num_samples,
        transform,
    )


if __name__ == "__main__":
    main()
