import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

BASE_MODEL = "allenai/OLMo-1B"
LORA_PATH = "./olmo-1b-bbbp-lora"
VAL_CSV = "bbbp_val.csv"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"":0},
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

val_df  = pd.read_csv(VAL_CSV)

def predict_yes_probability(prompt: str) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    probs = torch.softmax(logits, dim=1)

    p_yes = probs[0, yes_id].item()
    p_no = probs[0, no_id].item()

    return p_yes/ (p_yes+p_no)

scores = []
labels = []

for _, row in tqdm(val_df.iterrows, total=len(val_df)):
    prompt = (
        "Is the following molecule BBB permeable?\n"
        f"SMILES: {row['smiles']}\n"
        "Answer:"
    )

    score = predict_yes_probability(prompt)

    scores.append(score)
    labels.append(row["label"])

roc_auc = roc_auc_score(labels, scores)
print(f"ROC-AUC: {roc_auc:.4f}")