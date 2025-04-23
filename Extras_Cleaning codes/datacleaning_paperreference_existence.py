import pandas as pd

# Load the paper dataset
papers_df = pd.read_csv("Dataset/paper_cleaned.csv")
# Load the paper reference dataset
paper_ref_df = pd.read_csv("Dataset/paper_reference_cleaned.csv")

# Strip spaces from column names for consistency
papers_df.columns = papers_df.columns.str.strip()
paper_ref_df.columns = paper_ref_df.columns.str.strip()

# Ensure Paper IDs are strings for both DataFrames
papers_df["Paper ID"] = papers_df["Paper ID"].astype(str).str.strip()
paper_ref_df["Paper ID"] = paper_ref_df["Paper ID"].astype(str).str.strip()
paper_ref_df["Referenced Paper ID"] = paper_ref_df["Referenced Paper ID"].astype(str).str.strip()

# Check that all Paper IDs and Referenced Paper IDs in paper_reference.csv exist in paper.csv
valid_papers = papers_df["Paper ID"].unique()

# Filter the paper_reference.csv to keep only valid references
valid_references = paper_ref_df[
    paper_ref_df["Paper ID"].isin(valid_papers) & 
    paper_ref_df["Referenced Paper ID"].isin(valid_papers)
]

# # Optional: Flag or inspect invalid references
# invalid_references = paper_ref_df[
#     ~paper_ref_df["Paper ID"].isin(valid_papers) | 
#     ~paper_ref_df["Referenced Paper ID"].isin(valid_papers)
# ]

# if not invalid_references.empty:
#     print("⚠️ Warning: Invalid references found (referenced papers not in paper.csv):")
#     print(invalid_references)

# Save the cleaned paper_reference.csv with valid references
valid_references.to_csv("Dataset/paper_reference_cleaned2.csv", index=False)

print("✅ Citation chains intact. Cleaned paper_reference.csv saved.")
