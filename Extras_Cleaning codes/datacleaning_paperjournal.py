import pandas as pd

# Load the CSV
df = pd.read_csv("Dataset/paper_journal.csv")

# Strip whitespace from column names (just in case)
df.columns = df.columns.str.strip()

# Ensure all columns exist before cleaning
expected_cols = ["Paper ID", "Journal Name", "Journal Publisher"]
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
else:
    # Convert all columns to string and strip whitespaces
    for col in expected_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows with missing Paper ID or Journal Name (but allow null Publisher)
    df = df[df["Paper ID"].notna() & df["Journal Name"].notna()]

    # Drop duplicates if any
    df = df.drop_duplicates()

    # Save cleaned version
    df.to_csv("Dataset/paper_journal_cleaned.csv", index=False)
    print("✅ paper_journal.csv cleaned and saved as paper_journal_cleaned.csv")
