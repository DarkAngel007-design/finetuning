import deepchem as dc
import pandas as pd
from format_instructions import format_example

def load_bbbp():
    tasks, datasets, transformers = dc.molnet.load_bbbp(
        featurizer = "Raw",
        split = "scaffold"
    )
    train, valid, test = datasets
    return train, valid, test

def to_dataframe (dc_dataset):
    return pd.DataFrame({
        "smiles": dc_dataset.X,
        "label": dc_dataset.y.flatten().astype(int)
    })

if __name__ =="__main__":
    train, val, test  = load_bbbp()
    
    train_df = to_dataframe(train)
    val_df = to_dataframe(val)

    train_df["text"] = train_df.apply(format_example, axis=1)
    val_df["text"] = val_df.apply(format_example, axis=1)

    train_df[["text"]].to_csv("bbbp_train_instruct.csv", index = False)
    val_df[["text"]].to_csv("bbbp_val_instruct.csv", index = False)


    print("Saved BBBP instruction CSVs")
