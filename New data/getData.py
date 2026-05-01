import pandas as pd
import numpy as np
import glob
import os

# ---- CONFIG ----
data_folder = r"C:\Users\User\Documents\GitHub\Spheroids\New data"

val_initial_cells = [3375, 1000]


def parse_path(filepath):
    base = os.path.basename(filepath).replace(".csv", "")
    parts = base.split("_")

    try:
        # Results_1000_2.csv
        if len(parts) >= 3 and parts[0] == "Results":
            initial_cells = int(parts[1])
            day = int(parts[2])
            return initial_cells, day

        # fallback: folder name contains initial cells
        folder = os.path.basename(os.path.dirname(filepath))
        initial_cells = int(folder)
        day = int(parts[-1])
        return initial_cells, day

    except Exception as e:
        print("⚠️ Could not parse:", filepath, "->", parts, "Error:", e)
        return None, None


# ---- FIND FILES ----
all_files = (
    glob.glob(os.path.join(data_folder, "Results_*.csv")) +
    glob.glob(os.path.join(data_folder, "*", "Results_*.csv"))
)

print("Found files:", len(all_files))
for f in all_files[:10]:
    print("  ", f)

if len(all_files) == 0:
    raise FileNotFoundError(
        "No Results_*.csv files found. Check data_folder path."
    )


# ---- LOAD FILES ----
all_data = []

for f in all_files:
    init, day = parse_path(f)

    if init is None:
        continue

    df = pd.read_csv(f)

    # clean column names
    df.columns = [str(c).strip() for c in df.columns]

    print("\nReading:", f)
    print("Columns:", list(df.columns))

    if "Area" not in df.columns:
        print("⚠️ File has no 'Area' column:", f)
        continue

    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df = df.dropna(subset=["Area"])

    if df.empty:
        print("⚠️ No valid numeric Area values:", f)
        continue

    df["Radius"] = np.sqrt(df["Area"] / np.pi)

    all_data.append({
        "InitialCells": init,
        "Day": day,
        "MeanArea": df["Area"].mean(),
        "StdArea": df["Area"].std(),
        "MeanRadius": df["Radius"].mean(),
        "StdRadius": df["Radius"].std(),
        "N": df["Radius"].count()
    })


# ---- AGGREGATE ----
if len(all_data) == 0:
    raise ValueError(
        "Files were found, but none produced valid data. "
        "Check whether the column is really named 'Area'."
    )

agg = pd.DataFrame(all_data)
agg = agg.sort_values(["InitialCells", "Day"])


# ---- SPLIT ----
train_set = agg[~agg["InitialCells"].isin(val_initial_cells)].copy()
val_set = agg[agg["InitialCells"].isin(val_initial_cells)].copy()


# ---- SAVE ----
train_set.to_csv("train_dataset.csv", index=False)
val_set.to_csv("val_dataset.csv", index=False)

print("\n✅ Training set:", train_set.shape)
print("✅ Validation set:", val_set.shape)

print("\nPreview:")
print(agg.head())