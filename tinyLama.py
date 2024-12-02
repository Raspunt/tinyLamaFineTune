import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


class ModelManager:
    def __init__(self, model_id: str, save_path: str):
        self.model_id = model_id
        self.save_path = save_path

    def load_tokenizer_and_model(self, pretrained_path=None):
        print("Loading tokenizer and model...")
        path = pretrained_path if pretrained_path else self.model_id
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu", trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    @staticmethod
    def configure_lora():
        print("Configuring LoRA...")
        return LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )


class Trainer:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def train_model(self, data_path, epochs=10, learning_rate=2e-4):
        print("Loading dataset for training...")
        train_dataset = load_dataset("json", data_files=data_path, split="train")
        print(f"Training dataset size: {len(train_dataset)} records.")

        tokenizer, model = self.model_manager.load_tokenizer_and_model()
        peft_config = self.model_manager.configure_lora()
        model = get_peft_model(model, peft_config)

        

        training_args = TrainingArguments(
            output_dir=self.model_manager.save_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            optim="adamw_torch",
            logging_steps=20,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            num_train_epochs=epochs,
            save_strategy="epoch",
            dataloader_num_workers=2,
            report_to="none"
        )

        print("Initializing trainer...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
            peft_config=peft_config,
        )

        print("Starting training...")
        trainer.train()
        trainer.save_model(self.model_manager.save_path)
        print(f"Training complete. Model saved to '{self.model_manager.save_path}'.")

    def fine_tune_model(self, data_path, checkpoint_path, epochs=5, learning_rate=2e-5):
        print("Loading dataset for fine-tuning...")
        train_dataset = load_dataset("json", data_files=data_path, split="train")
        print(f"Fine-tuning dataset size: {len(train_dataset)} records.")

        tokenizer, model = self.model_manager.load_tokenizer_and_model(pretrained_path=checkpoint_path)
        peft_config = self.model_manager.configure_lora()
        model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(
            output_dir=self.model_manager.save_path,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            optim="adamw_torch",
            logging_steps=20,
            learning_rate=learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="linear",
            num_train_epochs=epochs,
            save_strategy="epoch",
            dataloader_num_workers=2,
            report_to="none"
        )

        print("Initializing trainer for fine-tuning...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
            peft_config=peft_config,
        )

        print("Starting fine-tuning...")
        trainer.train()
        trainer.save_model(self.model_manager.save_path)
        print(f"Fine-tuning complete. Model saved to '{self.model_manager.save_path}'.")


class TextGenerator:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate_text(self, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
        print("Loading model and tokenizer for text generation...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path)

        model.to("cpu")
        print("Tokenizing input text...")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")

        print("Generating text...")
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text


class Utils:
    @staticmethod
    def get_last_checkpoint(save_path):
        print("Getting the last checkpoint...")
        checkpoints = [
            int(ckpt.split("-")[1])
            for ckpt in os.listdir(save_path)
            if ckpt.startswith("checkpoint")
        ]
        return f"{save_path}/checkpoint-{max(checkpoints)}" if checkpoints else None
