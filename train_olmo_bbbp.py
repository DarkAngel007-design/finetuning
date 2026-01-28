import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_name = "allenai/OLMo-7B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast = False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = {"": 0},
    torch_dtype = torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.config.use_cache = False

dataset = load_dataset(
    "csv",
    data_files={
        "train": "bbbp_train_instruct.csv",
        "validation": "bbbp_val_instruct.csv"
    }
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


training_args = TrainingArguments(
    output_dir="./olmo-bbbp",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    max_steps=50,
    logging_steps=10,
    save_steps=50,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    ddp_find_unused_parameters=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("olmo-bbbp-lora")
tokenizer.save_pretrained("olmo-bbbp-lora")











