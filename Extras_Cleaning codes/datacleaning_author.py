import pandas as pd

# Load the CSV file
authors_df = pd.read_csv("Dataset/author.csv", low_memory=False)

# Display initial info
print("Initial shape:", authors_df.shape)

# Strip whitespace for Author Name and URL
authors_df['Author Name'] = authors_df['Author Name'].astype(str).str.strip()
authors_df['Author URL'] = authors_df['Author URL'].astype(str).str.strip()

# Drop rows with missing Author ID or Author Name (these are essential)
authors_df = authors_df.dropna(subset=['Author ID', 'Author Name'])

# Convert Author ID to string (to avoid merge issues)
authors_df['Author ID'] = authors_df['Author ID'].astype(str).str.strip()

# Drop exact duplicate rows (same ID, Name, URL)
authors_df = authors_df.drop_duplicates()

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
