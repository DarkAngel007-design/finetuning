import pandas as pd
from sklearn.model_selection import train_test_split
from format_instructions import format_example


BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"


def load_bbbp_raw():
    df = pd.read_csv(BBBP_URL)

    # Keep only what we need
    df = df[["smiles", "p_np"]].dropna()
    df["label"] = df["p_np"].astype(int)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    val_df, _ = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"]
    )

    return train_df, val_df


if __name__ == "__main__":
    train_df, val_df = load_bbbp_raw()

    train_df["text"] = train_df.apply(format_example, axis=1)
    val_df["text"] = val_df.apply(format_example, axis=1)

    train_df[["text"]].to_csv("bbbp_train_instruct.csv", index=False)
    val_df[["text"]].to_csv("bbbp_val_instruct.csv", index=False)

    print("Saved BBBP instruction CSVs")
