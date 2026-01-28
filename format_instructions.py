import pandas as pd

def format_example(smiles, label):
    label_text = "Permeable" if label == 1 else "Not permeable"

    return{
        "text": f"""### Instruction:
Given thte SMILES string of a molecule, determine whether it can cross the blood-brain barrier.print

### Input:
SMILES:{smiles}

### Output:
{label_text}"""
    }

def process (csv_in, csv_out):
    df = pd.read_csv(csv_in)
    formatted = [format_example(r.smiles, r.label) for r in df.itertuples()]
    out_df = pd.DataFrame(formatted)
    out_df.to_csv(csv_out, index=False)

if __name__ =="__main__":
    process("bbbp_train.csv", "bbbp_train_instruct.csv")
    process("bbbp_val.csv", "bbbp_val_instruct.csv")