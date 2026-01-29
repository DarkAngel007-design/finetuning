import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


model_name = "allenai/OLMo-1B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast = False,
    trust_remote_code=True
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map = {"": 0},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# model.gradient_checkpointing_enable()
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
    output_dir="./olmo-1b-bbbp-qlora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    max_steps=50,
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing=True,
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
model.save_pretrained("./olmo-1b-bbbp-qlora")
tokenizer.save_pretrained("./olmo-1b-bbbp-qlora")
trainer.save_model("./olmo-1b-bbbp-qlora")

















