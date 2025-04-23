import pandas as pd

# Load the raw file
df = pd.read_csv("Dataset/paper_topic.csv")

# Rename columns to strip any unnecessary whitespace
df.columns = df.columns.str.strip()

# Drop rows with any nulls in Paper ID or Topic ID
df = df.dropna(subset=["Paper ID", "Topic ID"])

# Convert Paper ID and Topic ID to string (to avoid numeric inconsistencies)
df["Paper ID"] = df["Paper ID"].astype(str).str.strip()
df["Topic ID"] = df["Topic ID"].astype(str).str.strip()

# Drop duplicate paper-topic pairs
df = df.drop_duplicates(subset=["Paper ID", "Topic ID"])

# Save cleaned version
df.to_csv("Dataset/paper_topic_cleaned.csv", index=False)

print("✅ paper_topic.csv cleaned and saved as paper_topic_cleaned.csv")
