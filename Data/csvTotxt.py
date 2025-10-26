import pandas as pd
import os

# ---- CONFIG ----
train_file = "train_dataset.csv"
val_file = "val_dataset.csv"

output_train = "train_txt"
output_val = "val_txt"

# create folders if not exist
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_val, exist_ok=True)

# ---- STEP 1: Load datasets ----
train = pd.read_csv(train_file)
val = pd.read_csv(val_file)

# ---- STEP 2: Save one file per InitialCells ----
def save_txt_per_group(df, out_folder):
    for init_cells, subdf in df.groupby("InitialCells"):
        out_path = os.path.join(out_folder, f"{init_cells}.txt")
        with open(out_path, "w") as f:
            f.write(f"# InitialCells = {init_cells}\n")
            f.write("# Day\tMeanRadius\tStdRadius\tN\n")
            for _, row in subdf.sort_values("Day").iterrows():
                f.write(f"{int(row.Day)}\t{row.MeanRadius:.3f}\t{row.StdRadius:.3f}\t{int(row.N)}\n")

# apply
save_txt_per_group(train, output_train)
save_txt_per_group(val, output_val)

print("✅ Exported .txt files to", output_train, "and", output_val)
