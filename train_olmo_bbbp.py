import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

model_name = "allenai/OLMo-7B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast = False,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map = "auto",
    torch_dtype = torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qkv_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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
        max_length=256
    )

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

training_args = TrainingArguments(
    output_dir="./olmo-bbbp",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_steps=50,
    logging_steps=10,
    save_steps=50,
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

trainer.train()




