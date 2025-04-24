import pandas as pd

# Load the raw file
df = pd.read_csv("Dataset/topic.csv")

# Rename columns to strip any unnecessary whitespace
df.columns = df.columns.str.strip()

# Drop rows with any nulls in Topic ID or Topic Name (keep Topic URL)
df = df.dropna(subset=["Topic ID", "Topic Name"])

# Convert Topic ID to string (to avoid numeric inconsistencies)
df["Topic ID"] = df["Topic ID"].astype(str).str.strip()

# Strip whitespace from string fields
df["Topic Name"] = df["Topic Name"].astype(str).str.strip()
df["Topic URL"] = df["Topic URL"].astype(str).str.strip()

# Drop duplicate topic entries by Topic ID
df = df.drop_duplicates(subset=["Topic ID"])

# Save cleaned version
df.to_csv("Dataset/topic_cleaned.csv", index=False)

print("✅ topic.csv cleaned and saved as topic_cleaned.csv")
