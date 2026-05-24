import pandas as pd
import os

# ---- CONFIG ----
train_file = "train_dataset.csv"
val_file = "val_dataset.csv"

output_train = "train_txt"
output_val = "val_txt"

# Create folders if not exist
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_val, exist_ok=True)


# ---- STEP 1: Load datasets ----
train = pd.read_csv(train_file)
val = pd.read_csv(val_file)


# ---- STEP 2: Save one file per InitialCells ----
def save_txt_per_group(df, out_folder):
    """
    Saves one txt file per InitialCells value.

    Each file contains all available columns except InitialCells.
    Usually this includes:
    Day, N,
    MeanArea, StdArea,
    MeanRadius, StdRadius,
    MeanFeret, StdFeret,
    MeanCircularity, StdCircularity,
    etc.
    """

    if "InitialCells" not in df.columns:
        raise ValueError("Dataset must contain 'InitialCells' column.")

    if "Day" not in df.columns:
        raise ValueError("Dataset must contain 'Day' column.")

    for init_cells, subdf in df.groupby("InitialCells"):
        subdf = subdf.sort_values("Day").copy()

        # Export all columns except InitialCells
        export_columns = [c for c in subdf.columns if c != "InitialCells"]

        out_path = os.path.join(out_folder, f"{int(init_cells)}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# InitialCells = {init_cells}\n")
            f.write("# " + "\t".join(export_columns) + "\n")

            for _, row in subdf.iterrows():
                values = []

                for col in export_columns:
                    value = row[col]

                    if pd.isna(value):
                        values.append("nan")
                    elif col in ["Day", "N"]:
                        values.append(str(int(value)))
                    else:
                        values.append(f"{value:.6g}")

                f.write("\t".join(values) + "\n")


# ---- STEP 3: Apply ----
save_txt_per_group(train, output_train)
save_txt_per_group(val, output_val)

print("✅ Exported .txt files to", output_train, "and", output_val)