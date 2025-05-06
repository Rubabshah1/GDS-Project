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
from sklearn.metrics import roc_auc_score, roc_curve
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
            RETURN elementId(p) AS paperId, elementId(j) AS journalId, 1 AS label
        """)
        train_existing_links = pd.DataFrame([dict(record) for record in result])

        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE toInteger(toString(y.year)) >= 2015
            RETURN elementId(p) AS paperId, elementId(j) AS journalId, 1 AS label
        """)
        test_existing_links = pd.DataFrame([dict(record) for record in result])

        # Negative examples (papers not published in journals but related by topics)
        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) < 2015
            RETURN elementId(p) AS paperId, elementId(j) AS journalId, 0 AS label
        """)
        # result = session.run("""
        #     MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)
        #     MATCH (j:Journal)
        #     MATCH path = (p)-[:PUBLISHED_IN*2..3]-(j)
        #     WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) < 2015
        #     RETURN DISTINCT p.id AS paperId, j.id AS journalId, 0 AS label
        # """)
        train_missing_links = pd.DataFrame([dict(record) for record in result]).drop_duplicates()

        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) >= 2015
            RETURN elementId(p) AS paperId, elementId(j) AS journalId, 0 AS label
        """)
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
            CALL gds.graph.drop('paperJournalGraph')
        """)
        session.run("""
            CALL gds.graph.project(
                'paperJournalGraph',
                ['Paper', 'Journal', 'Topic', 'Field_of_study'],
                {
                    HAS_TOPIC: {type: 'HAS_TOPIC', orientation: 'UNDIRECTED'},
                    WORKS_ON: {type: 'WORKS_ON', orientation: 'UNDIRECTED'},
                    PUBLISHED_IN: {type: 'PUBLISHED_IN', orientation: 'UNDIRECTED'},
                    CITES: {type: 'CITES', orientation: 'UNDIRECTED'}
                }
            )
        """)
        # Debug: Check graph projection
        result = session.run("CALL gds.graph.list('paperJournalGraph') YIELD nodeCount, relationshipCount")
        # graph_info = [dict(record) for record in result]
        # print("Graph Projection Info:", graph_info)

        # Debug: Check relationships
        # result = session.run("""
        #     MATCH ()-[:HAS_TOPIC|WORKS_ON|PUBLISHED_IN]-()
        #     RETURN count(*) AS relCount
        # """)
        # rel_count = [dict(record) for record in result]
        # print("Relationship Count:", rel_count)

        # Debug: Test GDS with sample nodes
        # result = session.run("""
        #     MATCH (p:Paper), (j:Journal)
        #     WHERE elementId(p) = 1 AND elementId(j) = 1
        #     RETURN gds.alpha.linkprediction.adamicAdar(p, j, {relationshipQuery: 'CITES|HAS_TOPIC|WORKS_ON|PUBLISHED_IN'}) AS aa,
        #            gds.alpha.linkprediction.preferentialAttachment(p, j, {relationshipQuery: 'CITES|HAS_TOPIC|WORKS_ON|PUBLISHED_IN'}) AS pa,
        #            gds.alpha.linkprediction.commonNeighbors(p, j, {relationshipQuery: 'CITES|HAS_TOPIC|WORKS_ON|PUBLISHED_IN'}) AS cn
        # """)
        # gds_sample = [dict(record) for record in result]
        # print("Sample GDS Scores:", gds_sample)

    # Compute Adamic/Adar, Preferential Attachment, and Common Neighbors using GDS
    query_graph_features = """
    UNWIND $pairs AS pair
    MATCH (p:Paper), (j:Journal) WHERE elementId(p) = pair.paperId and elementId(j) = pair.journalId
    RETURN pair.paperId AS paperId,
           pair.journalId AS journalId,
           gds.alpha.linkprediction.adamicAdar(p, j) AS aa,
           gds.alpha.linkprediction.preferentialAttachment(p, j) AS pa,
           gds.alpha.linkprediction.commonNeighbors(p, j) AS cn
    """
    pairs = [{"paperId": row['paperId'], "journalId": row['journalId']} for _, row in data.iterrows()]
    with driver.session(database="neo4j") as session:
        result = session.run(query_graph_features, {"pairs": pairs})
        graph_features = pd.DataFrame([dict(record) for record in result])
        print("Sample Graph Features:", graph_features.head())  # Debug output

    # Compute Topic Overlap, Field of Study Match, and Journal Popularity
    query_additional_features = """
    UNWIND $pairs AS pair
    MATCH (p:Paper) WHERE elementId(p) = pair.paperId
    MATCH (j:Journal) WHERE elementId(j) = pair.journalId
    WITH pair, p, j
    OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j)
    WITH pair, p, j, count(DISTINCT t) AS topicOverlap
    OPTIONAL MATCH (p)-[:WORKS_ON]->(f:Field_of_study)<-[:WORKS_ON]-(p2:Paper)-[:PUBLISHED_IN]->(j)
    WITH pair, p, j, topicOverlap, count(DISTINCT f) AS field_match
    MATCH (j)<-[:PUBLISHED_IN]-(p2:Paper)
    RETURN pair.paperId AS paperId, pair.journalId AS journalId,
           topicOverlap, field_match, count(p2) AS journalPopularity
    """
    with driver.session(database="neo4j") as session:
        result = session.run(query_additional_features, {"pairs": pairs})
        additional_features = pd.DataFrame([dict(record) for record in result])

    # Merge features
    features = pd.merge(graph_features, additional_features, on=["paperId", "journalId"], how='left')
    return pd.merge(data, features, on=["paperId", "journalId"])

# classifier = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0)
# end::create-classifier[]

def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred)
        ]
    }
    
    if y_prob is not None:
        metrics["Measure"].append("AUC-ROC")
        metrics["Score"].append(roc_auc_score(y_true, y_prob))
    
    return pd.DataFrame(metrics)

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

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

# Main execution
if __name__ == "__main__":
    
    # Section 3: Training and Evaluation
    # Prepare data and compute features
    # Prepare data
    train_df, test_df = prepare_paper_journal_data()
    train_features = apply_paper_journal_features(train_df, "before_2015")
    test_features = apply_paper_journal_features(test_df, "2015_and_later")

    # Save the feature-engineered datasets
    train_features.to_csv("Link Prediction/df_train_under_paper_journal.csv", index=False)
    test_features.to_csv("Link Prediction/df_test_under_paper_journal.csv", index=False)
    # Feature engineering
    
    # Define feature columns
    feature_cols = [
        'aa',
        'cn',
        'pa',
        'topicOverlap',
        'field_match',
        'journalPopularity'
    ]
    
    # Train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(train_features[feature_cols], train_features['label'])
    
    # Evaluate
    test_pred = clf.predict(test_features[feature_cols])
    test_prob = clf.predict_proba(test_features[feature_cols])[:, 1]
    
    print("Model Evaluation:")
    print(evaluate_model(test_features['label'], test_pred, test_prob))
    
    # Visualizations
    plot_roc_curve(test_features['label'], test_prob)
    feature_importance(feature_cols, clf)
    
    # Clean up
    driver.close()
