import pandas as pd

# Load the raw file
df = pd.read_csv("Dataset/journal.csv")

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Strip whitespace from values
df["Journal Name"] = df["Journal Name"].astype(str).str.strip()
df["Journal Publisher"] = df["Journal Publisher"].astype(str).str.strip()

# Drop rows only if Journal Name is missing
df = df.dropna(subset=["Journal Name"])

# Drop duplicates considering both columns
df = df.drop_duplicates(subset=["Journal Name", "Journal Publisher"])

# Save cleaned version
df.to_csv("Dataset/journal_cleaned.csv", index=False)

print("✅ journal.csv cleaned (publisher optional) and saved as journal_cleaned.csv")
