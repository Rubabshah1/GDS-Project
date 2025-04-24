```python
import pandas as pd
import hashlib
import uuid
import re
from datetime import datetime
import os
import logging
import unicodedata

# Setup logging for errors
logging.basicConfig(filename='cleaning_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(message)s')

# Directory for input and output CSVs
input_dir = '/Users/rubabshah/Desktop/gds/GDS-Project/Dataset/'
output_dir = './cleaned/'
os.makedirs(output_dir, exist_ok=True)

# Helper functions
def generate_id(row, fields):
    """Generate a unique ID based on specified fields."""
    unique_string = ''.join(str(row[field]) for field in fields if pd.notna(row[field]))
    return hashlib.sha1(unique_string.encode('utf-8')).hexdigest()

def standardize_date(date_str):
    """Convert date to YYYY-MM-DD format, return as string."""
    if pd.isna(date_str):
        return 'Unknown'
    try:
        return pd.to_datetime(date_str, errors='coerce').strftime('%Y-%m-%d')
    except:
        return 'Unknown'

def standardize_author_name(text):
    """Standardize author names: handle initials, hyphens, diacritics, and case."""
    if pd.isna(text) or text.strip() == '':
        return 'Unknown'
    
    # Normalize diacritics (e.g., José → Jose)
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    
    # Split into parts (handles spaces and hyphens)
    parts = text.strip().split()
    standardized_parts = []
    
    for part in parts:
        if '-' in part:
            hyphenated = '-'.join(p.capitalize() for p in part.split('-'))
            standardized_parts.append(hyphenated)
        else:
            if len(part) == 1 or (len(part) == 2 and part.endswith('.')):
                part = part.rstrip('.').capitalize() + '.'
            else:
                part = part.capitalize()
            standardized_parts.append(part)
    
    return ' '.join(standardized_parts)

# 1. Clean paper.csv
def clean_paper_csv(df):
    paper_id_pattern = re.compile(r'^paper[0-9]+$')
    invalid_ids = df['Paper ID'].str.match(paper_id_pattern, na=False)
    deleted_ids = set(df[invalid_ids]['Paper ID'].dropna())
    
    df = df[~invalid_ids]
    
    duplicate_ids = df[df['Paper ID'].duplicated(keep='first')]['Paper ID']
    deleted_ids.update(duplicate_ids)
    df = df.drop_duplicates(subset=['Paper ID'], keep='first')
    
    df['Paper DOI'] = df['Paper DOI'].fillna('NULL')
    df['Paper Title'] = df['Paper Title'].apply(standardize_author_name).fillna('Unknown')
    df['Paper Year'] = df['Paper Year'].astype(str).fillna('Unknown')
    df['Paper Citation Count'] = pd.to_numeric(df['Paper Citation Count'], errors='coerce').fillna(0).astype(int)
    df['Fields of Study'] = df['Fields of Study'].str.capitalize().fillna('Unknown')
    df['Journal Volume'] = df['Journal Volume'].fillna('Unknown')
    df['Journal Date'] = df['Journal Date'].apply(standardize_date)
    
    df = df.drop(columns=['Paper URL'])
    
    df = df.drop_duplicates(subset=['Paper Title', 'Paper Year', 'Paper DOI'], keep='first')
    
    df.to_csv(f'{output_dir}paper_cleaned.csv', index=False, sep='\t')
    return df, deleted_ids

# 2. Clean authors.csv
def clean_authors_csv(df):
    try:
        df['Author ID'] = df['Author ID'].astype(str).str.strip().str.lower()
        df['Author ID'] = df['Author ID'].apply(lambda x: str(uuid.uuid4()) if pd.isna(x) or x == '' else x)
        
        df['Author Name'] = df['Author Name'].apply(standardize_author_name)
        
        df['Author URL'] = df['Author URL'].fillna('NULL').str.strip()
        
        duplicate_ids = df[df['Author ID'].duplicated(keep='first')]['Author ID']
        if not duplicate_ids.empty:
            logging.warning(f"Removed {len(duplicate_ids)} duplicate Author IDs: {list(duplicate_ids)}")
        df = df.drop_duplicates(subset=['Author ID'], keep='first')
        
        df.to_csv(f'{output_dir}authors_cleaned.csv', index=False, sep='\t')
        return df
    except Exception as e:
        logging.error(f"Error cleaning authors.csv: {e}")
        raise

# 3. Clean author_paper.csv
def clean_author_paper_csv(paper_ids, author_ids, df):
    try:
        df = df[df['PaperID'].isin(paper_ids) & df['AuthorID'].isin(author_ids)]
        
        df = df.drop_duplicates(subset=['AuthorID', 'PaperID'], keep='first')
        
        df.to_csv(f'{output_dir}author_paper_cleaned.csv', index=False, sep='\t')
        return df
    except Exception as e:
        logging.error(f"Error cleaning author_paper.csv: {e}")
        raise

# 4. Clean topic.csv
def clean_topic_csv(df):
    df['Topic ID'] = df['Topic ID'].apply(lambda x: str(uuid.uuid4()) if pd.isna(x) else x)
    df['Topic Name'] = df['Topic Name'].apply(standardize_author_name).fillna('Unknown')
    df['Topic URL'] = df['Topic URL'].fillna('NULL')
    
    df = df.drop_duplicates(subset=['Topic ID'], keep='first')
    
    df.to_csv(f'{output_dir}topic_cleaned.csv', index=False, sep='\t')
    return df

# 5. Clean paper_topic.csv
def clean_paper_topic_csv(paper_ids, topic_ids, df):
    df = df[df['Paper ID'].isin(paper_ids) & df['Topic ID'].isin(topic_ids)]
    
    df = df.drop_duplicates(subset=['Paper ID', 'Topic ID'], keep='first')
    
    df.to_csv(f'{output_dir}paper_topic_cleaned.csv', index=False, sep='\t')
    return df

# 6. Clean paper_reference.csv
def clean_paper_reference_csv(paper_ids, deleted_ids, df):
    df = df[~df['Paper ID'].isin(deleted_ids) & ~df['Referenced Paper ID'].isin(deleted_ids)]
    
    df = df[df['Paper ID'].isin(paper_ids) & df['Referenced Paper ID'].isin(paper_ids)]
    
    df = df[df['Paper ID'] != df['Referenced Paper ID']]
    
    df = df.drop_duplicates(subset=['Paper ID', 'Referenced Paper ID'], keep='first')
    
    df.to_csv(f'{output_dir}paper_reference_cleaned.csv', index=False, sep='\t')
    return df

# 7. Clean journal.csv
def clean_journal_csv(df):
    try:
        # Lowercase and strip
        df['Journal Name'] = df['Journal Name'].str.lower().str.strip()
        df['Journal Publisher'] = df['Journal Publisher'].str.lower().str.strip()
        
        # Replace & with and
        df['Journal Name'] = df['Journal Name'].str.replace('&', 'and')
        df['Journal Publisher'] = df['Journal Publisher'].str.replace('&', 'and')
        
        # Fill missing publishers
        df['Journal Publisher'] = df['Journal Publisher'].fillna('unknown')
        
        # Extract email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        def extract_email_and_publisher(publisher):
            if pd.isna(publisher) or publisher == 'unknown':
                return 'unknown', 'Unknown'
            publisher = str(publisher)
            email_match = re.search(email_pattern, publisher)
            email = email_match.group(0) if email_match else 'Unknown'
            clean_publisher = re.sub(email_pattern, '', publisher).strip()
            clean_publisher = re.sub(r'\d+.*?(?=$|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', '', clean_publisher).strip()
            clean_publisher = clean_publisher.strip(',. ')
            return clean_publisher if clean_publisher else 'unknown', email
        
        df[['Journal Publisher', 'Email Address']] = df['Journal Publisher'].apply(
            lambda x: pd.Series(extract_email_and_publisher(x))
        )
        
        # Apply clustering rules for deduplication
        def apply_clustering_rules(df):
            df['Priority'] = df.apply(
                lambda x: 2 if x['Journal Publisher'] != 'unknown' and x['Email Address'] != 'Unknown' else 
                          1 if x['Email Address'] != 'Unknown' else 0, axis=1
            )
            result = []
            journal_counts = df['Journal Name'].value_counts()
            for name, group in df.groupby('Journal Name'):
                if journal_counts[name] == 1:
                    result.append(group)
                else:
                    complete_rows = group[(group['Journal Publisher'] != 'unknown') & (group['Email Address'] != 'Unknown')]
                    if not complete_rows.empty:
                        result.append(complete_rows.sort_values(by='Priority', ascending=False).head(1))
                    else:
                        for publisher, pub_group in group.groupby('Journal Publisher'):
                            if len(pub_group) > 1:
                                email_rows = pub_group[pub_group['Email Address'] != 'Unknown']
                                if not email_rows.empty:
                                    result.append(email_rows.sort_values(by='Priority', ascending=False).head(1))
                                else:
                                    result.append(pub_group.head(1))
                            else:
                                result.append(pub_group)
            return pd.concat(result).drop(columns=['Priority'])
        
        df = apply_clustering_rules(df)
        
        # Final deduplication
        df['Email_Priority'] = df['Email Address'].apply(lambda x: 0 if x == 'Unknown' else 1)
        df = df.sort_values(by=['Journal Name', 'Journal Publisher', 'Email_Priority'], ascending=[True, True, False])
        df = df.drop_duplicates(subset=['Journal Name', 'Journal Publisher'], keep='first')
        df = df.drop(columns=['Email_Priority'])
        
        # Log duplicates
        duplicate_names = df[df['Journal Name'].duplicated(keep=False)]
        if not duplicate_names.empty:
            logging.warning(f"Remaining duplicate Journal Names: {len(duplicate_names)} rows")
            logging.warning(duplicate_names[['Journal Name', 'Journal Publisher', 'Email Address']].to_string())
        
        df.to_csv(f'{output_dir}journal_cleaned.csv', index=False, sep='\t')
        return df
    except Exception as e:
        logging.error(f"Error cleaning journal.csv: {e}")
        raise

# 8. Clean paper_journal.csv
def clean_paper_journal_csv(paper_ids, journal_names, df):
    try:
        # Lowercase and strip
        df['Journal Name'] = df['Journal Name'].str.lower().str.strip()
        df['Journal Publisher'] = df['Journal Publisher'].str.lower().str.strip()
        
        # Replace & with and
        df['Journal Name'] = df['Journal Name'].str.replace('&', 'and')
        df['Journal Publisher'] = df['Journal Publisher'].str.replace('&', 'and')
        
        # Handle duplicates by keeping rows with non-missing publisher
        duplicate_mask = df.duplicated(subset=['Paper ID'], keep=False)
        duplicate_papers = df[duplicate_mask].sort_values(by='Paper ID')
        if not duplicate_papers.empty:
            logging.warning(f"Found {len(duplicate_papers['Paper ID'].unique())} duplicate Paper IDs")
            logging.warning(duplicate_papers[['Paper ID', 'Journal Name', 'Journal Publisher']].to_string())
            
            df['is_publisher_missing'] = df['Journal Publisher'].isna()
            df = df.sort_values(by=['Paper ID', 'is_publisher_missing'], ascending=[True, True])
            df = df.drop_duplicates(subset=['Paper ID'], keep='first').drop(columns=['is_publisher_missing'])
        
        # Fill missing publishers
        df['Journal Publisher'] = df['Journal Publisher'].fillna('unknown')
        
        # Extract email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        df['Email Address'] = df['Journal Publisher'].str.extract(f'({email_pattern})', expand=False)
        df['Journal Publisher'] = df['Journal Publisher'].str.replace(email_pattern, '', regex=True).str.strip(' ,.')
        
        # Filter valid Paper ID and Journal Name
        df = df[df['Paper ID'].isin(paper_ids) & df['Journal Name'].isin(journal_names)]
        
        # Check for remaining duplicates
        if df['Paper ID'].duplicated().any():
            logging.warning("Duplicate Paper IDs still exist in paper_journal.csv")
        
        df.to_csv(f'{output_dir}paper_journal_cleaned.csv', index=False, sep='\t')
        return df
    except Exception as e:
        logging.error(f"Error cleaning paper_journal.csv: {e}")
        raise

# Main function
def main():
    paper_dtypes = {
        'Paper ID': str,
        'Paper DOI': str,
        'Paper Title': str,
        'Paper Year': str,
        'Paper Citation Count': float,
        'Fields of Study': str,
        'Journal Volume': str,
        'Journal Date': str
    }
    author_dtypes = {
        'Author ID': str,
        'Author Name': str,
        'Author URL': str
    }
    author_paper_dtypes = {
        'AuthorID': str,
        'PaperID': str
    }
    topic_dtypes = {
        'Topic ID': str,
        'Topic Name': str,
        'Topic URL': str
    }
    paper_topic_dtypes = {
        'Paper ID': str,
        'Topic ID': str
    }
    paper_reference_dtypes = {
        'Paper ID': str,
        'Referenced Paper ID': str
    }
    journal_dtypes = {
        'Journal Name': str,
        'Journal Publisher': str
    }
    paper_journal_dtypes = {
        'Paper ID': str,
        'Journal Name': str,
        'Journal Publisher': str
    }

    try:
        paper_df = pd.read_csv(f'{input_dir}paper.csv', sep='\t', dtype=paper_dtypes, low_memory=False, on_bad_lines='warn')
        author_df = pd.read_csv(f'{input_dir}authors.csv', sep='\t', dtype=author_dtypes, low_memory=False, on_bad_lines='warn')
        author_paper_df = pd.read_csv(f'{input_dir}author_paper.csv', sep='\t', dtype=author_paper_dtypes, low_memory=False, on_bad_lines='warn')
        topic_df = pd.read_csv(f'{input_dir}topic.csv', sep='\t', dtype=topic_dtypes, low_memory=False, on_bad_lines='warn')
        paper_topic_df = pd.read_csv(f'{input_dir}paper_topic.csv', sep='\t', dtype=paper_topic_dtypes, low_memory=False, on_bad_lines='warn')
        paper_reference_df = pd.read_csv(f'{input_dir}paper_reference.csv', sep='\t', dtype=paper_reference_dtypes, low_memory=False, on_bad_lines='warn')
        journal_df = pd.read_csv(f'{input_dir}journal.csv', sep='\t', dtype=journal_dtypes, low_memory=False, on_bad_lines='warn')
        paper_journal_df = pd.read_csv(f'{input_dir}paper_journal.csv', sep='\t', dtype=paper_journal_dtypes, low_memory=False, on_bad_lines='warn')
        print("All CSVs loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading CSVs: {e}")
        raise

    paper_df, deleted_ids = clean_paper_csv(paper_df)
    paper_ids = set(paper_df['Paper ID'])
    
    author_df = clean_authors_csv(author_df)
    author_ids = set(author_df['Author ID'])
    
    author_paper_df = clean_author_paper_csv(paper_ids, author_ids, author_paper_df)
    
    topic_df = clean_topic_csv(topic_df)
    topic_ids = set(topic_df['Topic ID'])
    
    paper_topic_df = clean_paper_topic_csv(paper_ids, topic_ids, paper_topic_df)
    
    paper_reference_df = clean_paper_reference_csv(paper_ids, deleted_ids, paper_reference_df)
    
    journal_df = clean_journal_csv(journal_df)
    journal_names = set(journal_df['Journal Name'])
    
    paper_journal_df = clean_paper_journal_csv(paper_ids, journal_names, paper_journal_df)
    
    citation_counts = paper_reference_df.groupby('Referenced Paper ID').size().reindex(
        paper_df['Paper ID'], fill_value=0
    )
    paper_df['Inferred Citation Count'] = paper_df['Paper ID'].map(citation_counts)
    discrepancies = paper_df[
        paper_df['Paper Citation Count'] != paper_df['Inferred Citation Count']
    ]
    if not discrepancies.empty:
        print(f"Found {len(discrepancies)} citation count discrepancies.")
        paper_df['Paper Citation Count'] = paper_df['Inferred Citation Count']
    paper_df = paper_df.drop(columns=['Inferred Citation Count'])
    paper_df.to_csv(f'{output_dir}paper_cleaned.csv', index=False, sep='\t')
    
    print("Cleaning complete. Cleaned CSVs saved in", output_dir)
    print(f"Deleted Paper IDs: {len(deleted_ids)}")

if __name__ == '__main__':
    main()
```
