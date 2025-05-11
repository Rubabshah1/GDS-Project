import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import uuid

plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "admin"))

# Section 1: Train-Test Splitting
def prepare_paper_journal_data():
    with driver.session(database="neo4j") as session:
        # Positive examples
        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE toInteger(toString(y.year)) < 2016
            RETURN elementId(p) AS paperId, elementId(j) AS journalId, 1 AS label
        """)
        train_existing_links = pd.DataFrame([dict(record) for record in result])

        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:PUBLISHED_IN]->(j:Journal)
            WHERE toInteger(toString(y.year)) >= 2016
            RETURN elementId(p) AS paperId, elementId(j) AS journalId, 1 AS label
        """)
        test_existing_links = pd.DataFrame([dict(record) for record in result])

        # Negative examples
        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j:Journal), (p2)-[]-(y2:Year)
            WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) < 2016 AND toInteger(toString(y2.year)) < 2016 
            RETURN elementId(p) AS paperId, elementId(j) AS journalId, 0 AS label
        """)
        train_missing_links = pd.DataFrame([dict(record) for record in result]).drop_duplicates()

        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j:Journal), (p2)-[]-(y2:Year)
            WHERE NOT (p)-[:PUBLISHED_IN]->(j) AND toInteger(toString(y.year)) >= 2016 AND toInteger(toString(y2.year)) >= 2016 
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

        # Validate no overlapping pairs
        train_pairs = set(df_train[['paperId', 'journalId']].itertuples(index=False, name=None))
        test_pairs = set(df_test[['paperId', 'journalId']].itertuples(index=False, name=None))
        overlap = train_pairs.intersection(test_pairs)
        print(f"Overlap between train and test: {len(overlap)} pairs.")

        # Validate temporal split
        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)
            WHERE elementId(p) IN $paperIds
            RETURN elementId(p) AS paperId, y.year AS year
        """, {"paperIds": df_train['paperId'].tolist()})
        train_years = pd.DataFrame([dict(record) for record in result])
        print("Training set years:", len(train_years))

        result = session.run("""
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)
            WHERE elementId(p) IN $paperIds
            RETURN elementId(p) AS paperId, y.year AS year
        """, {"paperIds": df_test['paperId'].tolist()})
        test_years = pd.DataFrame([dict(record) for record in result])
        print("Test set years:", len(test_years))

        return df_train, df_test

# Section 2: Feature Engineering
def apply_paper_journal_features(data, year_filter):
    with driver.session(database="neo4j") as session:
        # Drop existing graph
        try:
            session.run("CALL gds.graph.drop('paperJournalGraph') YIELD graphName")
        except:
            print("No existing graph to drop.")

        # Project graph using updated GDS syntax
        year_condition = "WHERE y.year < 2016" if year_filter == "before_2016" else "WHERE y.year >= 2016"
        session.run(f"""
                    CALL gds.graph.project.cypher(
                'paperJournalGraph',
                'MATCH (n) WHERE n:Paper OR n:Journal OR n:Topic OR n:Field_of_study RETURN id(n) AS id, labels(n) AS labels',
                'MATCH (p:Paper)-[r:PUBLISHED_IN]->(j:Journal)<-[:PUBLISHED_IN]-(p2:Paper)-[:PUBLISHED_IN_YEAR]-(y:Year)
                 {year_condition}
                 RETURN id(p) AS source, id(j) AS target, type(r) AS type
                 UNION
                 MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[r:HAS_TOPIC]-(t:Topic)
                 {year_condition}
                 RETURN id(p) AS source, id(t) AS target, type(r) AS type
                 UNION
                 MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p:Paper)-[r:WORKS_ON]-(f:Field_of_study)
                 {year_condition}
                 RETURN id(p) AS source, id(f) AS target, type(r) AS type
                 UNION
                 MATCH (p1:Paper)-[:PUBLISHED_IN_YEAR]->(y1:Year),
                (p2:Paper)-[:PUBLISHED_IN_YEAR]->(y2:Year),
                (p1)-[r:CITES]->(p2)
                WHERE y1.year {'< 2016' if year_filter == 'before_2016' else '>= 2016'} AND y2.year {'< 2016' if year_filter == 'before_2016' else '>= 2016'}
                RETURN id(p1) AS source, id(p2) AS target, type(r) AS type'
            )
        """)

            # Compute centrality features (Degree, PageRank, Betweenness)
        try:
            # PageRank
            session.run("""
                CALL gds.pageRank.write('paperJournalGraph', {
                    maxIterations: 20,
                    dampingFactor: 0.85,
                    writeProperty: 'pagerank'
                })
                YIELD writeMillis
            """)

        except Exception as e:
            print(f"Error computing centrality features: {e}")
            return None

        # Compute graph-based link prediction features
        query_graph_features = """
            UNWIND $pairs AS pair
            MATCH (p:Paper), (j:Journal) WHERE elementId(p) = pair.paperId AND elementId(j) = pair.journalId
            RETURN pair.paperId AS paperId,
                   pair.journalId AS journalId,
                   gds.alpha.linkprediction.adamicAdar(p, j, {graphName: 'paperJournalGraph'}) AS aa,
                   gds.alpha.linkprediction.preferentialAttachment(p, j, {graphName: 'paperJournalGraph'}) AS pa,
                   gds.alpha.linkprediction.commonNeighbors(p, j, {graphName: 'paperJournalGraph'}) AS cn,
                   gds.alpha.linkprediction.resourceAllocation(p, j, {graphName: 'paperJournalGraph'}) AS ra,
                   gds.alpha.linkprediction.totalNeighbors(p, j, {graphName: 'paperJournalGraph'}) AS total_neighbors,
                   p.pageRank AS paperPageRank,
                   j.pageRank AS journalPageRank
        """
        pairs = [{"paperId": row['paperId'], "journalId": row['journalId']} for _, row in data.iterrows()]
        try:
            result = session.run(query_graph_features, {"pairs": pairs})
            graph_features = pd.DataFrame([dict(record) for record in result])
            print("Sample Graph Features:", graph_features.head())
        except Exception as e:
            print(f"Error computing graph features: {e}")
            return None

        # Compute additional features
        query_additional_features = f"""
            UNWIND $pairs AS pair
            MATCH (p:Paper) WHERE elementId(p) = pair.paperId
            MATCH (j:Journal) WHERE elementId(j) = pair.journalId
            MATCH (y:Year)-[:PUBLISHED_IN_YEAR]-(p)
            WHERE y.year {'< 2016' if year_filter == 'before_2016' else '>= 2016'}
            WITH pair, p, j
            OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(p2:Paper)-[:PUBLISHED_IN]->(j), (p2)-[:PUBLISHED_IN_YEAR]-(y2:Year)
            WHERE y2.year {'< 2016' if year_filter == 'before_2016' else '>= 2016'}
            WITH pair, p, j, count(DISTINCT t) AS topicOverlap
            OPTIONAL MATCH (p)-[:WORKS_ON]->(f:Field_of_study)<-[:WORKS_ON]-(p2:Paper)-[:PUBLISHED_IN]->(j), (p2)-[:PUBLISHED_IN_YEAR]-(y2:Year)
            WHERE y2.year {'< 2016' if year_filter == 'before_2016' else '>= 2016'}
            WITH pair, p, j, topicOverlap, count(DISTINCT f) AS field_match
            MATCH (j)<-[:PUBLISHED_IN]-(p2:Paper)-[:PUBLISHED_IN_YEAR]-(y2:Year)
            WHERE y2.year {'< 2016' if year_filter == 'before_2016' else '>= 2016'}
            return pair.paperId as paperId, pair.journalId as journalId, topicOverlap, field_match, count(p2) AS journalPopularity
        """
        try:
            result = session.run(query_additional_features, {"pairs": pairs})
            additional_features = pd.DataFrame([dict(record) for record in result])
        except Exception as e:
            print(f"Error computing additional features: {e}")
            return None

        # Merge features
        features = pd.merge(graph_features, additional_features, on=["paperId", "journalId"], how='left')
        return pd.merge(data, features, on=["paperId", "journalId"])

# Evaluation functions
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

def plot_roc_curve(y_true, y_prob, classifier_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{classifier_name} ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

def feature_importance(columns, classifier, classifier_name):
    if hasattr(classifier, 'feature_importances_'):
        df = pd.DataFrame({
            "Feature": columns,
            "Importance": classifier.feature_importances_
        })
        df = df.sort_values("Importance", ascending=False)
        ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
        ax.xaxis.set_label_text("")
        plt.title(f'Feature Importance - {classifier_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{classifier_name}.png')
        plt.close()

def compare_classifiers(metrics_dict):
    df = pd.concat([pd.DataFrame(metrics).assign(Classifier=name) for name, metrics in metrics_dict.items()])
    df_pivot = df.pivot(index='Classifier', columns='Measure', values='Score')
    ax = df_pivot.plot(kind='bar', figsize=(10, 6))
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('classifier_comparison.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Prepare data
    train_df, test_df = prepare_paper_journal_data()
    train_features = apply_paper_journal_features(train_df, "before_2016")
    if train_features is None:
        print("Failed to compute training features, exiting.")
        driver.close()
        exit(1)
    
    test_features = apply_paper_journal_features(test_df, "2016_and_later")
    if test_features is None:
        print("Failed to compute test features, exiting.")
        driver.close()
        exit(1)

    # Save datasets
    train_features.to_csv("Link Prediction/df_train_under_paper_journal.csv", index=False)
    test_features.to_csv("Link Prediction/df_test_under_paper_journal.csv", index=False)

    # Define feature columns
    feature_cols = [
        'aa', 'cn', 'pa', 'ra', 'total_neighbors',
        'paperPageRank', 'journalPageRank',
        'topicOverlap', 'field_match', 'journalPopularity'
        
    ]

    # Impute NaN values
    imputer = SimpleImputer(strategy='median')
    
    train_features[feature_cols] = imputer.fit_transform(train_features[feature_cols])
    test_features[feature_cols] = imputer.transform(test_features[feature_cols])

    # Initialize classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }

    metrics_dict = {}
    plt.figure(figsize=(8, 6))

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(train_features[feature_cols], train_features['label'])
        test_pred = clf.predict(test_features[feature_cols])
        test_prob = clf.predict_proba(test_features[feature_cols])[:, 1] if hasattr(clf, 'predict_proba') else None
        # test_prob = clf.predict_proba(test_features[feature_cols])[:, 1]
        # test_pred = (test_prob > 0.6).astype(int)
        metrics_dict[name] = evaluate_model(test_features['label'], test_pred, test_prob)
        print(f"\nEvaluation for {name}:")
        print(metrics_dict[name])
        if test_prob is not None:
            plot_roc_curve(test_features['label'], test_prob, name)
        feature_importance(feature_cols, clf, name) if hasattr(clf, 'feature_importances_') else None

    plt.savefig('roc_curve_all_classifiers.png')
    plt.close()

    # Compare classifiers
    compare_classifiers(metrics_dict)

    # Clean up
    driver.close()
