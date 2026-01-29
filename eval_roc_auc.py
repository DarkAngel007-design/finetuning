import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

BASE_MODEL = "allenai/OLMo-1B"
LORA_PATH = "./olmo-1b-bbbp-qlora"
VAL_CSV = "bbbp_val_labels.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

val_df = pd.read_csv(VAL_CSV)

YES_ID = tokenizer.encode(" Yes", add_special_tokens=False)[0]
NO_ID  = tokenizer.encode(" No",  add_special_tokens=False)[0]

@torch.no_grad()
def predict_yes_probability(smiles: str) -> float:
    prompt = (
        "You are a chemistry expert.\n\n"
        "Task: Predict whether the following molecule can cross the "
        "blood-brain barrier (BBB).\n\n"
        f"SMILES: {smiles}\n"
        "Answer (Yes or No):"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    logits = model(**inputs).logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    p_yes = probs[0, YES_ID].item()
    p_no  = probs[0, NO_ID].item()
    return p_yes / (p_yes + p_no)


scores, labels = [], []

for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
    scores.append(predict_yes_probability(row["smiles"]))
    labels.append(int(row["label"]))

roc_auc = roc_auc_score(labels, scores)
print(f"\n ROC-AUC (QLoRA): {roc_auc:.4f}")
