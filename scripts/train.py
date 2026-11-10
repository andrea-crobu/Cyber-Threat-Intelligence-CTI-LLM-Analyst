"""Phase 2: LoRA fine-tuning of the base model on the prepared CTI dataset."""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from config import RunConfig, parse_args
from model_utils import count_trainable_parameters, load_model, load_tokenizer, wrap_with_lora
from prepare_dataset import load_prepared


def build_data_collator(tokenizer) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )


def build_training_arguments(config: RunConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(config.output_dir / "training"),
        report_to="none",
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        seed=config.seed,
        save_strategy="no",
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
    )


def build_trainer(config: RunConfig, model, tokenizer, train_dataset, eval_dataset) -> Trainer:
    lora_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(lora_parameters, lr=config.learning_rate)
    return Trainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=build_data_collator(tokenizer),
        optimizers=(optimizer, None),
    )


def run_training(config: RunConfig) -> Path:
    if not config.dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {config.dataset_path}; run prepare_dataset.py first"
        )

    set_seed(config.seed)
    prepared = load_prepared(config.dataset_path)
    train_dataset = Dataset.from_dict(prepared["train"])
    eval_dataset = Dataset.from_dict(prepared["validation"])

    tokenizer = load_tokenizer(config)
    base_model = load_model(config, for_training=True)
    peft_model = wrap_with_lora(base_model, config)
    print(count_trainable_parameters(peft_model))

    trainer = build_trainer(config, peft_model, tokenizer, train_dataset, eval_dataset)
    trainer.train()

    config.peft_model_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(config.peft_model_path)
    tokenizer.save_pretrained(config.peft_model_path)
    print(f"Adapters saved to {config.peft_model_path}")
    return config.peft_model_path


def main(argv: list[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
