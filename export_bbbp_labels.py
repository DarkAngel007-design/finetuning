import deepchem as dc
import pandas as pd

def load_bbbp():
    tasks, datasets, transformers = dc.molnet.load_bbbp(
        featurizer="Raw",
        split="scaffold"
    )

    train, val, test = datasets
    return train,val

def to_df(dc_dataset):
    return pd.DataFrame({
        "smiles": dc_dataset.X,
        "label": dc_dataset.y_flatten().astype(int)
    })

if __name__ == "__main__":
    _, val = load_bbbp()
    val_df = to_df(val)
    val_df.to_csv("bbbp_val.csv", index=False)
    print("Saved bbbp_val.csv")
