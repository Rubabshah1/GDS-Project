import pandas as pd

# Load the raw file
df = pd.read_csv("Dataset/author_paper.csv")

# Rename columns to strip any unnecessary whitespace
df.columns = df.columns.str.strip()

# Drop rows with any nulls in Author ID or Paper ID
df = df.dropna(subset=["Author ID", "Paper ID"])

# Convert Author ID and Paper ID to string (to avoid numeric inconsistencies)
df["Author ID"] = df["Author ID"].astype(str).str.strip()
df["Paper ID"] = df["Paper ID"].astype(str).str.strip()

# Drop duplicate author-paper pairs
df = df.drop_duplicates(subset=["Author ID", "Paper ID"])

# Save cleaned version
df.to_csv("Dataset/author_paper_cleaned.csv", index=False)

print("✅ author_paper.csv cleaned and saved as author_paper_cleaned.csv")
