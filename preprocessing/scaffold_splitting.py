import random
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Load dataset (modify filenames if needed)
train_file = "/gscratch/aims/mingyulu/cell_painting/label_data/bray_2017/split_data/datasplit1-train.csv"
val_file = "/gscratch/aims/mingyulu/cell_painting/label_data/bray_2017/split_data/datasplit1-val.csv"
test_file = "/gscratch/aims/mingyulu/cell_painting/label_data/bray_2017/split_data/datasplit1-test.csv"

# Assuming CSVs contain a column "SMILES" for molecular structures
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

# Combine all molecules into one dataframe for scaffold-based splitting
df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

# Function to compute scaffold for a given SMILES
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    return None


# Compute scaffolds for unique SMILES
unique_smiles_df = df.drop_duplicates(subset=["SMILES"]).copy()
unique_smiles_df["Scaffold"] = unique_smiles_df["SMILES"].apply(get_scaffold)

# Group molecules by scaffold
scaffold_dict = defaultdict(list)
for idx, row in unique_smiles_df.iterrows():
    scaffold_dict[row["Scaffold"]].append(row["SMILES"])

# Convert scaffold dictionary to list of scaffold groups
scaffold_groups = list(scaffold_dict.values())

# Shuffle scaffolds to ensure randomness
random.seed(42)
random.shuffle(scaffold_groups)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Compute dataset sizes
total_scaffolds = len(scaffold_groups)
train_end = int(train_ratio * total_scaffolds)
val_end = train_end + int(val_ratio * total_scaffolds)

# Assign scaffolds to train, validation, and test sets
train_scaffolds = scaffold_groups[:train_end]
val_scaffolds = scaffold_groups[train_end:val_end]
test_scaffolds = scaffold_groups[val_end:]

# Flatten scaffold groups into SMILES lists
train_smiles = [smiles for group in train_scaffolds for smiles in group]
val_smiles = [smiles for group in val_scaffolds for smiles in group]
test_smiles = [smiles for group in test_scaffolds for smiles in group]

# Ensure all rows with the same SMILES stay together
train_df = df[df["SMILES"].isin(train_smiles)]
val_df = df[df["SMILES"].isin(val_smiles)]
test_df = df[df["SMILES"].isin(test_smiles)]

# Count unique SMILES in each dataset
num_unique_train = train_df["SMILES"].nunique()
num_unique_val = val_df["SMILES"].nunique()
num_unique_test = test_df["SMILES"].nunique()

# Print the results
print(f"Number of unique SMILES in Train set: {num_unique_train}")
print(f"Number of unique SMILES in Validation set: {num_unique_val}")
print(f"Number of unique SMILES in Test set: {num_unique_test}")

# Save new scaffold-based splits
train_df.to_csv(
    "/gscratch/aims/mingyulu/cell_painting/label_data/bray_2017/split_data/scaffold_train.csv",
    index=False,
)
val_df.to_csv(
    "/gscratch/aims/mingyulu/cell_painting/label_data/bray_2017/split_data/scaffold_val.csv",
    index=False,
)
test_df.to_csv(
    "/gscratch/aims/mingyulu/cell_painting/label_data/bray_2017/split_data/scaffold_test.csv",
    index=False,
)

print("Scaffold splitting completed! New datasets saved.")
