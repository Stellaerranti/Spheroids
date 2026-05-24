import pandas as pd
import numpy as np
import glob
import os

# ---- CONFIG ----
data_folder = r"C:\Users\User\Documents\GitHub\Spheroids\New data"

val_initial_cells = [3375,1000]


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
# ---- LOAD FILES ----
all_data = []

# ImageJ columns you want to use for fitting/comparison
parameter_columns = [
    "Area",
    "Mean",
    "StdDev",
    "Mode",
    "Min",
    "Max",
    "X",
    "Y",
    "XM",
    "YM",
    "Perim.",
    "Width",
    "Height",
    "Major",
    "Minor",
    "Angle",
    "Circ.",
    "Feret",
    "IntDen",
    "Median",
    "Skew",
    "Kurt",
    "RawIntDen",
    "MinFeret",
    "AR",
    "Round",
    "Solidity",
]

# Cleaner output names
name_map = {
    "Mean": "GrayMean",
    "StdDev": "GrayStdDev",
    "Mode": "GrayMode",
    "Min": "GrayMin",
    "Max": "GrayMax",
    "X": "CentroidX",
    "Y": "CentroidY",
    "XM": "CenterMassX",
    "YM": "CenterMassY",
    "Perim.": "Perimeter",
    "Circ.": "Circularity",
    "AR": "AspectRatio",
}

for f in all_files:
    init, day = parse_path(f)

    if init is None:
        continue

    df = pd.read_csv(f)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    print("\nReading:", f)
    print("Columns:", list(df.columns))

    if "Area" not in df.columns:
        print("⚠️ File has no 'Area' column:", f)
        continue

    # Keep only columns that exist in this file
    existing_columns = [c for c in parameter_columns if c in df.columns]

    # Convert all selected columns to numeric
    for col in existing_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Area is required because it defines whether a row is valid
    df = df.dropna(subset=["Area"])

    if df.empty:
        print("⚠️ No valid numeric Area values:", f)
        continue

    # Derived parameters
    df["Radius"] = np.sqrt(df["Area"] / np.pi)

    if "Major" in df.columns and "Minor" in df.columns:
        df["EllipseArea"] = np.pi * (df["Major"] / 2) * (df["Minor"] / 2)

    if "Area" in df.columns and "Perim." in df.columns:
        df["Circularity_calc"] = 4 * np.pi * df["Area"] / (df["Perim."] ** 2)

    if "Area" in df.columns and "Mean" in df.columns:
        df["IntDen_calc"] = df["Area"] * df["Mean"]

    derived_columns = [
        "Radius",
        "EllipseArea",
        "Circularity_calc",
        "IntDen_calc",
    ]

    all_columns = existing_columns + [
        c for c in derived_columns if c in df.columns
    ]

    row = {
        "InitialCells": init,
        "Day": day,
        "N": df["Area"].count(),
    }

    for col in all_columns:
        clean_name = name_map.get(col, col)
        clean_name = (
            clean_name.replace(".", "")
                      .replace("%", "Percent")
                      .replace(" ", "")
        )

        values = df[col].dropna()

        if values.empty:
            row[f"Mean{clean_name}"] = np.nan
            row[f"Std{clean_name}"] = np.nan
            continue

        row[f"Mean{clean_name}"] = values.mean()
        row[f"Std{clean_name}"] = values.std(ddof=1)

    all_data.append(row)


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