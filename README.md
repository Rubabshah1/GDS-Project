# GDS-Project-Spring-2025
CS343 Graph Data Science Project: Node Classification and Link Prediction in Bibliographic Data

Overview
This project aims to apply machine learning techniques to a bibliographic dataset using graph-based approaches. You will work in groups of up to four members. Each group must be divided into two sub-groups: one will be responsible for the node classification task, while the other will work on the link prediction task. The dataset is provided in CSV format, and students will need to clean, preprocess, and construct a graph to train models using Neo4j or Python-based machine learning frameworks integrated with Neo4j. The data cleaning, preprocessing, and graph construction should be common for both sub-groups.

Dataset Description
The dataset consists of the following CSV files:

#authors.csv: Contains Author ID (unique identifier), Author Name, Author URL
#journal.csv: Contains Journal Name, Journal Publisher
#paper.csv: Contains Paper ID (unique identifier), Paper DOI, Paper Title, Paper Year, Paper URL, Paper Citation Count, Field of Study, Journal Volume, Journal Date
#topic.csv: Contains Topic ID, Topic Name, Topic URL
#paper_journal.csv: Contains Paper ID, Journal Name, Journal Publisher
#paper_topic.csv: Contains Paper ID, Topic ID
#paper_reference.csv: Contains Paper ID, Referenced Paper ID
The dataset is available at GitHub Repository. You can learn more about the data and its extraction process in the following paper:

Rothenberger, Liane, Muhammad Qasim Pasta, and Daniel Mayerhoffer.
"Mapping and impact assessment of phenomenon-oriented research fields: The example of migration research."
Quantitative Science Studies 2, no. 4 (2021): 1466-1485.
