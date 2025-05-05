import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from neo4j import GraphDatabase

plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# tag::imports[]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
# end::imports[]

# Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Farzana123"
driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "admin"))

# Section 1: Train-Test Splitting
def prepare_paper_journal_data():
    with driver.session(database="neo4j") as session:
        # Positive examples (existing publications)
        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE toInteger(toString(y.year)) < 2015
            RETURN id(p) AS paperId, id(j) AS journalId, 1 AS label
        """)
        train_existing_links = pd.DataFrame([dict(record) for record in result])

        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE toInteger(toString(y.year)) >= 2015
            RETURN id(p) AS paperId, id(j) AS journalId, 1 AS label
        """)
        test_existing_links = pd.DataFrame([dict(record) for record in result])

        # Negative examples (papers not published in journals but related by topics)
        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) < 2015
            RETURN id(p) AS paperId, id(j) AS journalId, 0 AS label
        """)
        train_missing_links = pd.DataFrame([dict(record) for record in result]).drop_duplicates()

        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) >= 2015
            RETURN id(p) AS paperId, id(j) AS journalId, 0 AS label
        """)
        # result = session.run("""
        #     MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)
        #     MATCH (j:Journal)
        #     MATCH path = (p)-[:HAS_TOPIC|CITES|PUBLISHED_IN*2..3]-(j)
        #     WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) < 2015
        #     RETURN DISTINCT p.id AS paperId, j.id AS journalId, 0 AS label
        # """)
        test_missing_links = pd.DataFrame([dict(record) for record in result]).drop_duplicates()

        # Combine and downsample
        train_df = pd.concat([train_missing_links, train_existing_links], ignore_index=True)
        count_class_1 = train_df.label.value_counts()[1]
        df_class_0 = train_df[train_df['label'] == 0].sample(count_class_1, random_state=42)
        df_class_1 = train_df[train_df['label'] == 1]
        df_train = pd.concat([df_class_0, df_class_1], axis=0)

        test_df = pd.concat([test_missing_links, test_existing_links], ignore_index=True)
        count_class_1 = test_df.label.value_counts()[1]
        df_class_0 = test_df[test_df['label'] == 0].sample(count_class_1, random_state=42)
        df_class_1 = test_df[test_df['label'] == 1]
        df_test = pd.concat([df_class_0, df_class_1], axis=0)

        return df_train, df_test

# Section 2: Feature Engineering
def apply_paper_journal_features(data, year_filter):
    # Create a subgraph projection for GDS
    with driver.session(database="neo4j") as session:
        session.run("""
            CALL gds.graph.drop('paperJournalGraph', false)
        """)
        session.run("""
            CALL gds.graph.project(
                'paperJournalGraph',
                ['Paper', 'Journal', 'Topic', 'Field_of_study'],
                {
                    HAS_TOPIC: {type: 'HAS_TOPIC', orientation: 'UNDIRECTED'},
                    WORKS_ON: {type: 'WORKS_ON', orientation: 'UNDIRECTED'},
                    PUBLISHED_IN: {type: 'PUBLISHED_IN', orientation: 'UNDIRECTED'}
                    // CITES: {type: 'CITES', orientation: 'UNDIRECTED'}
                }
            )
        """)
        # Debug: Check graph projection
        result = session.run("CALL gds.graph.list('paperJournalGraph') YIELD nodeCount, relationshipCount")
        graph_info = [dict(record) for record in result]
        print("Graph Projection Info:", graph_info)

        # Debug: Check relationships
        result = session.run("""
            MATCH ()-[:HAS_TOPIC|WORKS_ON|PUBLISHED_IN]-()
            RETURN count(*) AS relCount
        """)
        rel_count = [dict(record) for record in result]
        print("Relationship Count:", rel_count)

        # Debug: Test GDS with sample nodes
        result = session.run("""
            MATCH (p:Paper), (j:Journal)
            WHERE id(p) = 1 AND id(j) = 1
            RETURN gds.alpha.linkprediction.adamicAdar(p, j, {relationshipQuery: 'HAS_TOPIC|WORKS_ON|PUBLISHED_IN'}) AS aa,
                   gds.alpha.linkprediction.preferentialAttachment(p, j, {relationshipQuery: 'HAS_TOPIC|WORKS_ON|PUBLISHED_IN'}) AS pa,
                   gds.alpha.linkprediction.commonNeighbors(p, j, {relationshipQuery: 'HAS_TOPIC|WORKS_ON|PUBLISHED_IN'}) AS cn
        """)
        gds_sample = [dict(record) for record in result]
        print("Sample GDS Scores:", gds_sample)

    # Compute Adamic/Adar, Preferential Attachment, and Common Neighbors using GDS
    query_graph_features = """
    UNWIND $pairs AS pair
    MATCH (p:Paper), (j:Journal) WHERE id(p) = pair.paperId and id(j) = pair.journalId
    RETURN pair.paperId AS paperId,
           pair.journalId AS journalId,
           gds.alpha.linkprediction.adamicAdar(p, j, {relationshipQuery: 'PUBLISHED_IN'}) AS aa,
           gds.alpha.linkprediction.preferentialAttachment(p, j, {relationshipQuery: 'PUBLISHED_IN'}) AS pa,
           gds.alpha.linkprediction.commonNeighbors(p, j, {relationshipQuery: 'PUBLISHED_IN'}) AS cn
    """
    pairs = [{"paperId": row['paperId'], "journalId": row['journalId']} for _, row in data.iterrows()]
    with driver.session(database="neo4j") as session:
        result = session.run(query_graph_features, {"pairs": pairs})
        graph_features = pd.DataFrame([dict(record) for record in result])
        print("Sample Graph Features:", graph_features.head())  # Debug output

    # Compute Topic Overlap, Field of Study Match, and Journal Popularity
    query_additional_features = """
    UNWIND $pairs AS pair
    MATCH (p:Paper) WHERE id(p) = pair.paperId
    MATCH (j:Journal) WHERE id(j) = pair.journalId
    WITH pair, p, j
    OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j)
    WITH pair, p, j, count(DISTINCT t) AS topicOverlap
    OPTIONAL MATCH (p)-[:WORKS_ON]->(f:Field_of_study)<-[:WORKS_ON]-(p2:Paper)-[:PUBLISHED_IN]->(j)
    WITH pair, p, j, topicOverlap, count(DISTINCT f) AS fieldOfStudyMatch
    MATCH (j)<-[:PUBLISHED_IN]-(p2:Paper)
    RETURN pair.paperId AS paperId, pair.journalId AS journalId,
           topicOverlap, fieldOfStudyMatch, count(p2) AS journalPopularity
    """
    with driver.session(database="neo4j") as session:
        result = session.run(query_additional_features, {"pairs": pairs})
        additional_features = pd.DataFrame([dict(record) for record in result])

    # Merge features
    features = pd.merge(graph_features, additional_features, on=["paperId", "journalId"], how='left')
    return pd.merge(data, features, on=["paperId", "journalId"])

# Section 3: Training and Evaluation
# Prepare data and compute features
df_train, df_test = prepare_paper_journal_data()
df_train = apply_paper_journal_features(df_train, "before_2015")
df_test = apply_paper_journal_features(df_test, "2015_and_later")

# Save the feature-engineered datasets
df_train.to_csv("Link Prediction/df_train_under_paper_journal.csv", index=False)
df_test.to_csv("Link Prediction/df_test_under_paper_journal.csv", index=False)

# Display sample data
print("Training Data Sample:")
print(df_train.sample(5))
print("\nTest Data Sample:")
print(df_test.sample(5))

# Create the random forest classifier
# tag::create-classifier[]
classifier = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0)
# end::create-classifier[]

# Train the model
# tag::train-model[]
columns = [
    "aa", "pa", "cn",  # graph-based features
    "topicOverlap", "fieldOfStudyMatch", "journalPopularity"  # additional features
]

X = df_train[columns]
y = df_train["label"]
classifier.fit(X, y)
# end::train-model[]

# Evaluate the model
# tag::evaluation-functions[]
def evaluate_model(predictions, actual):
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy_score(actual, predictions), 
                  precision_score(actual, predictions), 
                  recall_score(actual, predictions)]
    })

def feature_importance(columns, classifier):        
    print("Feature Importance")
    df = pd.DataFrame({
        "Feature": columns,
        "Importance": classifier.feature_importances_
    })
    df = df.sort_values("Importance", ascending=False)    
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()
# end::evaluation-functions[]

# Test the model
# tag::test-model[]
predictions = classifier.predict(df_test[columns])
y_test = df_test["label"]

eval_result = evaluate_model(predictions, y_test)
print("\nModel Evaluation:")
print(eval_result)
# end::test-model[]

# Save evaluation results
eval_result.to_csv("Link Prediction/model-eval-paper-journal.csv", index=False)

# Display feature importance
feature_importance(columns, classifier)

# Section 4: Predict New Links
def predict_links(paper_journal_pairs):
    # Prepare new data with the same features
    new_data = pd.DataFrame(paper_journal_pairs, columns=['paperId', 'journalId', 'label'])
    new_data = apply_paper_journal_features(new_data, "before_2015")
    
    # Predict probabilities
    X_new = new_data[columns]
    probabilities = classifier.predict_proba(X_new)[:, 1]  # Probability of class 1 (link exists)
    predictions = classifier.predict(X_new)  # Binary prediction (0 or 1)
    
    # Combine results
    new_data['probability'] = probabilities
    new_data['predicted_link'] = predictions
    return new_data[['paperId', 'journalId', 'probability', 'predicted_link']]

# Example usage: Check predicted links for new paper-journal pairs
new_paper_journal_pairs = [
    {'paperId': 1, 'journalId': 1, 'label': 0},
    {'paperId': 2, 'journalId': 2, 'label': 0}
]
# predicted_links = predict_links(new_paper_journal_pairs)
# print("\nNumber of Predicted Links:", len(predicted_links))
# print("Predicted Links with Probabilities:")
# print(predicted_links)

# Clean up GDS projection
with driver.session(database="neo4j") as session:
    session.run("""
        CALL gds.graph.drop('paperJournalGraph', false)
    """)

# Close Neo4j driver
driver.close()
