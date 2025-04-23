import pandas as pd

# Load the CSV
df = pd.read_csv("Dataset/paper_reference.csv", low_memory=False)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Convert both columns to string and strip whitespace
df["Paper ID"] = df["Paper ID"].astype(str).str.strip()
df["Referenced Paper ID"] = df["Referenced Paper ID"].astype(str).str.strip()

# Drop rows where Paper ID or Referenced Paper ID is missing
df = df[df["Paper ID"].notna() & df["Referenced Paper ID"].notna()]

# Remove self-citations (where Paper ID is equal to Referenced Paper ID)
df = df[df["Paper ID"] != df["Referenced Paper ID"]]

# Drop exact duplicate references
df = df.drop_duplicates()

# Save cleaned version
df.to_csv("Dataset/paper_reference_cleaned.csv", index=False)
print("✅ paper_reference.csv cleaned and saved as paper_reference_cleaned.csv")
