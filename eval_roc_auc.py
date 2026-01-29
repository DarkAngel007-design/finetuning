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

TEMPERATURE = 1.0

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

YES_VARIANTS = [" Yes", "Yes", " yes", "YES"]
NO_VARIANTS  = [" No", "No", " no", "NO"]

YES_IDS, NO_IDS = [], []

for v in YES_VARIANTS:
    ids = tokenizer.encode(v, add_special_tokens=False)
    if ids:
        YES_IDS.append(ids[0])

for v in NO_VARIANTS:
    ids = tokenizer.encode(v, add_special_tokens=False)
    if ids:
        NO_IDS.append(ids[0])

YES_IDS = list(set(YES_IDS))
NO_IDS  = list(set(NO_IDS))

print("YES token ids:", YES_IDS)
print("NO token ids :", NO_IDS)

def make_prompt(smiles: str) -> str:
    return f"""You are an expert chemist.

Task: Predict if the following molecule can cross the blood-brain barrier.

Examples:
SMILES: CCO
Answer: Yes

SMILES: CC(=O)Oc1ccccc1C(=O)O
Answer: No

Now predict:
SMILES: {smiles}
Answer:"""

@torch.no_grad()
def predict_yes_probability(smiles: str) -> float:
    prompt = make_prompt(smiles)

    inputs = tokenizer(
        prompt,
        return_tensors="pt", 
        truncation=True,
        max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)

    logits = model(**inputs).logits[:, -1, :]
    logits = logits/TEMPERATURE
    probs = torch.softmax(logits, dim=-1)

    p_yes = sum(probs[0, i].item() for i in YES_IDS if i < probs.shape[1])
    p_no  = sum(probs[0, i].item() for i in NO_IDS  if i < probs.shape[1])
    return p_yes / (p_yes + p_no + 1e-9)


scores, labels = [], []

for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
    scores.append(predict_yes_probability(row["smiles"]))
    labels.append(int(row["label"]))

roc_auc = roc_auc_score(labels, scores)
print(f"\n ROC-AUC (QLoRA): {roc_auc:.4f}")
