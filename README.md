# GDS-Project-Spring-2025

**CS343 Graph Data Science Project: Node Classification and Link Prediction in Bibliographic Data**

---

## Overview

This project applies machine learning techniques to a bibliographic dataset using graph-based approaches. The project is divided into two parts:

* **Node Classification**: Predict node labels (e.g., fields of study or publication types).
* **Link Prediction**: Predict future or missing connections between papers (e.g., citations).

Graph construction, preprocessing, and cleaning are **shared** between both subgroups.

---

## 📂 Dataset Description

The dataset is provided in CSV format and includes:

* `authors.csv`: Author ID, Name, URL
* `journal.csv`: Journal Name, Publisher
* `paper.csv`: Paper metadata: ID, DOI, Title, Year, Citation Count, etc.
* `topic.csv`: Topic ID, Name, URL
* `paper_journal.csv`: Mapping between papers and journals
* `paper_topic.csv`: Mapping between papers and topics
* `paper_reference.csv`: Citation links (Paper ID → Referenced Paper ID)

### 📖 Source

Rothenberger, Liane, et al.
*"Mapping and impact assessment of phenomenon-oriented research fields: The example of migration research."*
Quantitative Science Studies 2, no. 4 (2021): 1466–1485.

---

##  Reproducing the Results

Follow the steps below to reproduce the graph-based models:

---

### 1. Data Loading and Preprocessing

The Dataset_cleand folder contains all the cleaned csv and a file named loading.cypher for loading all data on neo4j and making relationships.

**File**: `loading.cypher`

* Use these Cypher queries to import CSV files into Neo4j.

```bash
# Run this in Neo4j Browser or via Python driver
:source loading.cypher
```

**Cleaning Scripts**: Located in `Extras_Cleaning codes/`

* Handles missing values, inconsistent formats, and irrelevant entries.
* Outputs cleaned versions saved in `Dataset_Cleaned/`.

---

### 2. 🔗 Graph Construction

* Nodes: Authors, Papers, Topics, Journals
* Relationships:

  * `(:Paper)-[:CITES]->(:Paper)`
  * `(:Paper)-[:HAS_TOPIC]->(:Topic)`
  * `(:Paper)-[:PUBLISHED_IN]->(:Journal)`
  * `(:Author)-[:AUTHORED]->(:Paper)`
  * `(:Author)-[:CO_AUTHOR]->(:AUTHOR)`
  * `(:Paper)-[:PUBLISHED_IN_YEAR]->(:YEAR)`
  * `(:Paper)-[:WORKS_ON]->(:Field_of_study)`

Graph construction is done via Neo4j and Cypher using cleaned data.

---

### 3. Model Training

* **Node Classification**

  * Directory: `Node Classification/`
  * Uses GNN-based models (e.g., GraphSAGE, GCN) to classify paper nodes.

* **Link Prediction**

  * Directory: `Link Prediction/`
  * Models trained to predict potential citation links.

Both tasks are run using Python with Neo4j integration via `neo4j` driver.

---

## Directory Structure

```bash
GDS-Project-Spring-2025/
│
├── Dataset_Original/           # Raw CSV files
├── Dataset_Cleaned/            # Preprocessed data
  ├── loading.cypher              # Cypher queries to load data into Neo4j
├── Extras_Cleaning codes/      # Python script for cleaning
│
├── Node Classification/        # Models, code, and results
├── Link Prediction/            # Models, code, and results
│
└── README.md                   # This file
```

---


