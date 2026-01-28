import torch
from transformers import (
    AutoTokenizer,
    AutoModelCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_datset
import bitsandbytes as bnb

model_name = "allenai/OLMo-7B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast = False
)

model = AutoModelCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map = "auto",
    torch_dytpe = torch.float16,
    low_cpu_mem_usage=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_datset(
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
        max_length=512
    )

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

training_args = TrainingArguments(
    output_dir="./olmo-bbbp",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=200,
    fp16=True,
    report_to="none"
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
