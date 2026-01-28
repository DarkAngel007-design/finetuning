import deepchem as dc
import pandas as pd
from deepchem.feat import RawFeaturizer

def load_bbbp():
    tasks, datasets, transformers = dc.molnet.load_bbbp(
        featurizer=RawFeaturizer(),   # REQUIRED but ignored
        split="scaffold",
        reload=True
    )
    train, val, test = datasets
    return val

if __name__ == "__main__":
    val = load_bbbp()

    # IMPORTANT: use ids, NOT X
    val_df = pd.DataFrame({
        "smiles": val.ids,
        "label": val.y.flatten().astype(int)
    })

    val_df.to_csv("bbbp_val.csv", index=False)
    print("Saved bbbp_val.csv")
