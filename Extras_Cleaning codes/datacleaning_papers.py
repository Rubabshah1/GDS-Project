import pandas as pd

# Load the CSV with low_memory=False to avoid warnings
df = pd.read_csv("Dataset/paper.csv", low_memory=False)

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Drop rows with missing essential fields (Paper ID and Paper Title)
df = df.dropna(subset=["Paper ID", "Paper Title"])

# Convert Paper ID to string and strip whitespace
df["Paper ID"] = df["Paper ID"].astype(str).str.strip()

# Clean string columns (updated with correct column names)
string_cols = [
    "Paper DOI", "Paper Title", 
    "Fields of Study", "Journal Volume", "Journal Date"
]
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Convert numeric columns
df["Paper Year"] = pd.to_numeric(df["Paper Year"], errors="coerce")

# Impute missing Citation Count as 0
df["Paper Citation Count"] = pd.to_numeric(df["Paper Citation Count"], errors="coerce").fillna(0).astype(int)

# Drop the 'Paper URL' column only
df = df.drop(columns=["Paper URL"], errors='ignore')

# Drop duplicate Paper IDs
df = df.drop_duplicates(subset=["Paper ID"])

# Save the cleaned version
df.to_csv("Dataset/paper_cleaned.csv", index=False)
print("✅ paper.csv cleaned and saved as paper_cleaned.csv")
