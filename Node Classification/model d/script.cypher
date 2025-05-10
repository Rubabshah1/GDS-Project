// Paper node
CREATE INDEX paper_id_index IF NOT EXISTS FOR (p:Paper) ON (p.ID);

// Author node
CREATE INDEX author_id_index IF NOT EXISTS FOR (a:Author) ON (a.ID);

// Journal node
CREATE INDEX journal_name_index IF NOT EXISTS FOR (j:Journal) ON (j.name);

// Topic node
CREATE INDEX topic_id_index IF NOT EXISTS FOR (t:Topic) ON (t.ID);



CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///paper_cleaned.csv' AS row FIELDTERMINATOR '\t' RETURN row",
    "MERGE (p:Paper {ID: row['Paper ID']})
     SET p.DOI = row['Paper DOI'],
         p.title = row['Paper Title'],
         p.year = row['Paper Year'],
         p.citationCount = toInteger(row['Paper Citation Count']),
         p.fieldsOfStudy = row['Fields of Study'],
         p.journalVol = row['Journal Volume'],
         p.journalDate = row['Journal Date']",
    {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///authors_cleaned.csv' AS row RETURN row",
    "MERGE (a:Author {ID: row['Author ID']})
     SET a.name = row['Author Name'],
         a.URL = row['Author URL']",
    {batchSize: 1000, parallel: false}
);

match(a:Author) set a.ID= toInteger(a.ID);

CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///journal_cleaned.csv' AS row  RETURN row",
    "MERGE (j:Journal {name: row['Journal Name']})
     SET j.publisher = row['Journal Publisher']",
    {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///topic_cleaned.csv' AS row RETURN row",
    "MERGE (t:Topic {ID: row['Topic ID']})
     SET t.name = row['Topic Name'],
         t.URL = row['Topic URL']",
    {batchSize: 1000, parallel: false}
);

match(t:Topic) set t.ID= toInteger(t.ID);

CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///author_paper_cleaned.csv' AS row RETURN row",
    "MATCH (a:Author {ID: toInteger(row['Author ID'])})
     MATCH (p:Paper {ID: row['Paper ID']})
     MERGE (a)-[:AUTHORED]->(p)",
    {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///paper_reference_cleaned.csv' AS row RETURN row",
    "MATCH (p1:Paper {ID: row['Paper ID']})
     MATCH (p2:Paper {ID: row['Referenced Paper ID']})
     MERGE (p1)-[:CITES]->(p2)",
    {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///paper_journal_cleaned.csv' AS row RETURN row",
    "MATCH (p:Paper {ID: row['Paper ID']})
     MATCH (j:Journal {name: row['Journal Name']})
     MERGE (p)-[:PUBLISHED_IN]->(j)",
    {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
    "LOAD CSV WITH HEADERS FROM 'file:///paper_topic_cleaned.csv' AS row RETURN row",
    "MATCH (p:Paper {ID: row['Paper ID']})
     MATCH (t:Topic {ID: toInteger(row['Topic ID'])})
     MERGE (p)-[:HAS_TOPIC]->(t)",
    {batchSize: 1000, parallel: false}
);



// CALL apoc.periodic.iterate(
//     "MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
//      WHERE id(a1) < id(a2)
//      RETURN a1, a2, p",
//     "MERGE (a1)-[:COAUTHORED]->(a2)",
//     {batchSize: 1000, parallel: false}
// );

// CALL apoc.periodic.iterate(
//     "MATCH (j:Journal)
//      RETURN j",
//     "MERGE (pub:Publisher {publisher: j.publisher})
//      MERGE (j)-[:PUBLISHED_BY]->(pub)",
//     {batchSize: 1000, parallel: false}
// );


CALL apoc.periodic.iterate(
    "MATCH (p:Paper)
     WHERE p.year IS NOT NULL
     WITH p, toInteger(p.year) AS yearInt
     WHERE yearInt IS NOT NULL
     RETURN p, yearInt",
    "MERGE (y:Year {year: yearInt})
     MERGE (p)-[:PUBLISHED_IN_YEAR]->(y)",
    {batchSize: 1000, parallel: false}
);

match (j:Journal) remove j.publisher;

MATCH (p:Paper)-[r:PUBLISHED_IN]->(j:Journal)
SET r.journalVol = toInteger(p.journalVol),
    r.journalDate = date(p.journalDate)
REMOVE p.journalVol, p.journalDate;

match (p:Paper) remove p.journalDate, p.journalVol;

CALL apoc.periodic.iterate(
    "MATCH (p:Paper) WHERE p.fieldsOfStudy IS NOT NULL AND p.fieldsOfStudy <> 'Unknown' RETURN p",
    "WITH p, split(replace(p.fieldsOfStudy, ';', ','), ',') AS fields
     UNWIND fields AS field
     MERGE (f:Field {name: toLower(trim(field))})",
    {batchSize: 1000, parallel: false}
);

CALL apoc.periodic.iterate(
    "MATCH (p:Paper) WHERE p.fieldsOfStudy IS NOT NULL AND p.fieldsOfStudy <> 'Unknown' RETURN p",
    "WITH p, split(replace(p.fieldsOfStudy, ';', ','), ',') AS fields
     UNWIND fields AS field
     MATCH (f:Field {name: toLower(trim(field))})
     MERGE (p)-[:HAS_FIELD]->(f)",
    {batchSize: 1000, parallel: false}
);


match(p:Paper) remove p.fieldsOfStudy;

match(f:field) set f.id=f(id)
same for journal


// Step 1: Assign the dominant fieldCategory to each Journal
MATCH (j:Journal)<-[:PUBLISHED_IN]-(p:Paper)-[:HAS_FIELD]->(f:Field)
WITH j, f.id AS fieldId, COUNT(p) AS paperCount
ORDER BY paperCount DESC
WITH j, COLLECT({fieldId: fieldId, count: paperCount}) AS fieldCounts
SET j.fieldCategory = fieldCounts[0].fieldId;

// Step 2: Verify the assignment
MATCH (j:Journal)
RETURN j.name, j.fieldCategory
LIMIT 5;

// Define the list of all 19 fields
WITH [
  'medicine',
  'political science',
  'sociology',
  'computer science',
  'history',
  'economics',
  'geography',
  'mathematics',
  'psychology',
  'philosophy',
  'art',
  'biology',
  'business',
  'engineering',
  'materials science',
  'physics',
  'chemistry',
  'geology',
  'environmental science'
] AS allFields

// Match all journals
MATCH (j:Journal)

// Unwind the list of fields to process each field for each journal
UNWIND allFields AS field

// Optionally match citations to papers in the specified field
OPTIONAL MATCH (j)<-[:PUBLISHED_IN]-(p:Paper)-[:CITES]->(c:Paper)-[:HAS_FIELD]->(f:Field)
WHERE f.name = field

// Aggregate citation counts (0 if no citations)
WITH j, field, COUNT(c) AS citationCount

// Set the citation count property for each field (replace spaces with underscores)
SET j[REPLACE(field, ' ', '_') + 'CitationCount'] = citationCount

// Return results for verification
RETURN j.name AS journal, 
       field, 
       citationCount
ORDER BY j.name, field;
