import pandas as pd

# Load the CSV file
authors_df = pd.read_csv("Dataset/author.csv", low_memory=False)

# Display initial info
print("Initial shape:", authors_df.shape)

# Strip whitespace and normalize case for Author Name
authors_df['Author Name'] = authors_df['Author Name'].astype(str).str.strip()

# Drop the 'Author URL' column as it's not needed for the classification task
authors_df = authors_df.drop(columns=['Author URL'], errors='ignore')

# Drop rows with missing Author ID or Author Name (these are essential)
authors_df = authors_df.dropna(subset=['Author ID', 'Author Name'])

# Convert Author ID to string (just in case it's numerical and causes merge issues later)
authors_df['Author ID'] = authors_df['Author ID'].astype(str).str.strip()

# Drop exact duplicate rows (same ID, Name)
authors_df = authors_df.drop_duplicates(subset=['Author ID', 'Author Name'])

# Check for duplicate Author IDs (may indicate conflicting data)
duplicate_ids = authors_df[authors_df.duplicated('Author ID', keep=False)]
if not duplicate_ids.empty:
    print("⚠️ Warning: Conflicting entries found for same Author ID:")
    print(duplicate_ids.sort_values('Author ID'))

    # Optionally, export them for manual review
    duplicate_ids.to_csv("conflicting_authors.csv", index=False)

# Save cleaned file
authors_df.to_csv("Dataset/authors_cleaned.csv", index=False)
print("✅ Cleaned authors.csv saved as authors_cleaned.csv")
