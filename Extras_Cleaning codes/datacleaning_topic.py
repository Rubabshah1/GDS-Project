import pandas as pd

# Load the raw file
df = pd.read_csv("Dataset/topic.csv")

# Rename columns to strip any unnecessary whitespace
df.columns = df.columns.str.strip()

# Drop rows with any nulls in Topic ID or Topic Name (Topic URL will be removed)
df = df.dropna(subset=["Topic ID", "Topic Name"])

# Convert Topic ID to string (to avoid numeric inconsistencies)
df["Topic ID"] = df["Topic ID"].astype(str).str.strip()

# Drop duplicate topic entries
df = df.drop_duplicates(subset=["Topic ID"])

# Drop the 'Topic URL' column as it's not needed
df = df.drop(columns=["Topic URL"], errors='ignore')

# Save cleaned version
df.to_csv("Dataset/topic_cleaned.csv", index=False)

print("✅ topic.csv cleaned and saved as topic_cleaned.csv")
