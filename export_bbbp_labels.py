import deepchem as dc
import pandas as pd

def load_bbbp():
    tasks, datasets, transformers = dc.molnet.load_bbbp(
        featurizer=None,      # 🔥 IMPORTANT FIX
        split="scaffold",
        reload=True
    )
    train, val, test = datasets
    return val

if __name__ == "__main__":
    val = load_bbbp()

    val_df = pd.DataFrame({
        "smiles": val.ids,                 # SMILES live in ids
        "label": val.y.flatten().astype(int)
    })

    val_df.to_csv("bbbp_val.csv", index=False)
    print("Saved bbbp_val.csv")
