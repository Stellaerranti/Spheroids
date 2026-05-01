import pandas as pd
import numpy as np
import glob
import os

# ---- CONFIG ----
# Path to the top-level folder that contains subfolders like 500, 1000, 15625 ...
data_folder = r"C:\Users\Дмитрий\Documents\GitHub\Spheroids\Spheroids\New data"
val_initial_cells = [3375, 1000]   # which groups to keep for validation

# ---- STEP 1: Helper to parse path ----
# Example: "...\500\Results_500_1.csv"
def parse_path(filepath):
    folder = os.path.basename(os.path.dirname(filepath))   # e.g. "500"
    base = os.path.basename(filepath).replace(".csv", "")  # "Results_500_1"
    parts = base.split("_")
    try:
        initial_cells = int(folder)    # from folder name
        day = int(parts[-1])           # last number in filename
        return initial_cells, day
    except Exception as e:
        print("⚠️ Could not parse:", filepath, "->", parts, "Error:", e)
        return None, None

# ---- STEP 2: Loop through files ----
all_data = []
for f in glob.glob(os.path.join(data_folder, "*", "Results_*.csv")):
    init, day = parse_path(f)
    if init is None:
        continue

    df = pd.read_csv(f)

    # drop the unnamed column if present
    if " " in df.columns:
        df = df.drop(columns=[" "])

    # check if Area exists
    if "Area" not in df.columns:
        print("⚠️ File has no 'Area' column:", f)
        continue

    # compute radius from area
    df["Radius"] = np.sqrt(df["Area"] / np.pi)

    # aggregate per file
    all_data.append({
        "InitialCells": init,
        "Day": day,
        "MeanRadius": df["Radius"].mean(),
        "StdRadius": df["Radius"].std(),
        "N": df["Radius"].count()
    })

agg = pd.DataFrame(all_data)

# ---- STEP 3: Split ----
train_set = agg[~agg["InitialCells"].isin(val_initial_cells)].copy()
val_set = agg[agg["InitialCells"].isin(val_initial_cells)].copy()

# ---- STEP 4: Save ----
train_set.to_csv("train_dataset.csv", index=False)
val_set.to_csv("val_dataset.csv", index=False)

print("✅ Training set:", train_set.shape)
print("✅ Validation set:", val_set.shape)
print("\nPreview of aggregated dataset:")
print(agg.head())
