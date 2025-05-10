import pandas as pd
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Initialize the Neo4j driver
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "mariaadnan"))

# Cypher query to retrieve features and target variable
query = """
MATCH (j:Journal)
WHERE j.field_category IS NOT NULL
OPTIONAL MATCH (j)<-[:PUBLISHED_IN]-(p:Paper)
WITH j, COUNT(p) AS paperCount
OPTIONAL MATCH (j)<-[:PUBLISHED_IN]-(p:Paper)-[:CITES]->(c:Paper)
WITH j, paperCount, COUNT(c) AS citationCount
RETURN j.name AS journal,
       j.field_category AS field_category,
       paperCount,
       CASE WHEN paperCount > 0 THEN toFloat(citationCount) / paperCount ELSE 0 END AS avgCitations,
       j.medicineCitationCount AS medicine,
       j.political_scienceCitationCount AS political_science,
       j.sociologyCitationCount AS sociology,
       j.computer_scienceCitationCount AS computer_science,
       j.historyCitationCount AS history,
       j.economicsCitationCount AS economics,
       j.geographyCitationCount AS geography,
       j.mathematicsCitationCount AS mathematics,
       j.psychologyCitationCount AS psychology,
       j.philosophyCitationCount AS philosophy,
       j.artCitationCount AS art,
       j.biologyCitationCount AS biology,
       j.businessCitationCount AS business,
       j.engineeringCitationCount AS engineering,
       j.materials_scienceCitationCount AS materials_science,
       j.physicsCitationCount AS physics,
       j.chemistryCitationCount AS chemistry,
       j.geologyCitationCount AS geology,
       j.environmental_scienceCitationCount AS environmental_science
ORDER BY j.name;
"""

# Run the query and store the results in a DataFrame
with driver.session() as session:
    result = session.run(query)
    df = pd.DataFrame([record.data() for record in result])

# Close driver
driver.close()

# Check the first few rows of the DataFrame
# print("DataFrame head:\n", df.head())

# Feature columns
feature_columns = [
    'paperCount',
    'avgCitations',
    'medicine',
    'political_science',
    'sociology',
    'computer_science',
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
    'materials_science',
    'physics',
    'chemistry',
    'geology',
    'environmental_science'
]

# Create normalized features
normalized_df = df[feature_columns].copy()
# Replace 0 max values with 1 to avoid division by zero
max_values = normalized_df.max()
max_values[max_values == 0] = 1
normalized_df = normalized_df / max_values

# Add journal and field_category
normalized_df['journal'] = df['journal']
normalized_df['field_category'] = df['field_category']

# Handle missing values
normalized_df.fillna(0, inplace=True)

# Merge similar classes
class_mapping = {
    'medicine': 'life_sciences',
    'biology': 'life_sciences',
    'sociology': 'social_sciences',
    'political science': 'social_sciences',
    'economics': 'social_sciences',
    'geography': 'social_sciences',
    'psychology': 'social_sciences',
    'physics': 'physical_sciences',
    'chemistry': 'physical_sciences',
    'materials science': 'physical_sciences',
    'geology': 'physical_sciences',
    'environmental science': 'physical_sciences',
    'history': 'humanities',
    'philosophy': 'humanities',
    'art': 'humanities',
    'mathematics': 'quantitative_fields',
    'computer science': 'quantitative_fields',
    'engineering': 'applied_fields',
    'business': 'applied_fields'
}
normalized_df['field_category'] = normalized_df['field_category'].replace(class_mapping)

# Remove classes with <3 samples
min_samples = 3
valid_classes = normalized_df['field_category'].value_counts()[normalized_df['field_category'].value_counts() >= min_samples].index
normalized_df = normalized_df[normalized_df['field_category'].isin(valid_classes)]

# Encode target variable
label_encoder = LabelEncoder()
normalized_df['field_category_encoded'] = label_encoder.fit_transform(normalized_df['field_category'])
field_categories = label_encoder.classes_

# Define features and target
X_normalized = normalized_df[feature_columns]
y = normalized_df['field_category_encoded']

# Scale features
scaler = StandardScaler()
X_normalized_scaled = scaler.fit_transform(X_normalized)

# Split data
X_train_normalized, X_test_normalized, y_train, y_test = train_test_split(X_normalized_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest with adjusted class weights
class_weights = {0: 2.8, 1: 1.0}  # Increase weight for life_sciences (0) to improve recall
clf_normalized_rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=2, random_state=42, class_weight=class_weights)
clf_normalized_rf.fit(X_train_normalized, y_train)

# Train HistGradientBoostingClassifier
clf_normalized_hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=5, min_samples_leaf=2, random_state=42)
clf_normalized_hgb.fit(X_train_normalized, y_train)

# Make predictions
y_pred_normalized_rf = clf_normalized_rf.predict(X_test_normalized)
y_pred_normalized_hgb = clf_normalized_hgb.predict(X_test_normalized)

# Decode predictions
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_normalized_rf_decoded = label_encoder.inverse_transform(y_pred_normalized_rf)
y_pred_normalized_hgb_decoded = label_encoder.inverse_transform(y_pred_normalized_hgb)

# Get unique classes in test set
unique_test_classes = np.unique(np.concatenate([y_test_decoded, y_pred_normalized_rf_decoded, y_pred_normalized_hgb_decoded]))
unique_test_labels = [field_categories[i] for i in np.unique(np.concatenate([y_test, y_pred_normalized_rf, y_pred_normalized_hgb]))]

# Evaluate Random Forest
print("Random Forest - Normalized Features:")
accuracy_normalized_rf = accuracy_score(y_test, y_pred_normalized_rf)
print("Accuracy:", accuracy_normalized_rf)
print(classification_report(y_test_decoded, y_pred_normalized_rf_decoded, labels=unique_test_classes, target_names=unique_test_labels, zero_division=0))

# Evaluate HistGradientBoostingClassifier
print("\nHistGradientBoostingClassifier - Normalized Features:")
accuracy_normalized_hgb = accuracy_score(y_test, y_pred_normalized_hgb)
print("Accuracy:", accuracy_normalized_hgb)
print(classification_report(y_test_decoded, y_pred_normalized_hgb_decoded, labels=unique_test_classes, target_names=unique_test_labels, zero_division=0))

# Cross-validation
cv_scores_normalized_rf = cross_val_score(clf_normalized_rf, X_normalized_scaled, y, cv=3)
cv_scores_normalized_hgb = cross_val_score(clf_normalized_hgb, X_normalized_scaled, y, cv=3)
print("\nRandom Forest - Normalized Features CV Scores:", cv_scores_normalized_rf, "Mean:", cv_scores_normalized_rf.mean())
print("HistGradientBoostingClassifier - Normalized Features CV Scores:", cv_scores_normalized_hgb, "Mean:", cv_scores_normalized_hgb.mean())

# Fine-tune Random Forest
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight=class_weights), param_grid_rf, cv=3)
grid_search_rf.fit(X_train_normalized, y_train)
print("\nRandom Forest - Best parameters:", grid_search_rf.best_params_)
print("Random Forest - Best score:", grid_search_rf.best_score_)
clf_normalized_rf = grid_search_rf.best_estimator_

# # Diagnostics
# print("\nNormalized Feature Stats (mean per field):")
# print(normalized_df[feature_columns].mean())
# print("\nClass Distribution:")
# print(normalized_df['field_category'].value_counts())
# print("\nMax Values for Normalization (check for zeros):")
# print(max_values)

# New data for prediction
new_data = pd.DataFrame({
    'paperCount': [500],
    'avgCitations': [2.5],
    'medicine': [120],
    'political_science': [0],
    'sociology': [0],
    'computer_science': [0],
    'history': [0],
    'economics': [0],
    'geography': [0],
    'mathematics': [0],
    'psychology': [0],
    'philosophy': [0],
    'art': [0],
    'biology': [0],
    'business': [0],
    'engineering': [0],
    'materials_science': [0],
    'physics': [0],
    'chemistry': [0],
    'geology': [0],
    'environmental_science': [0]
})
new_data_normalized = new_data / max_values
new_data_normalized.fillna(0, inplace=True)
new_data_normalized_scaled = scaler.transform(new_data_normalized)
prediction_normalized_rf = clf_normalized_rf.predict(new_data_normalized_scaled)
prediction_normalized_hgb = clf_normalized_hgb.predict(new_data_normalized_scaled)
print("Predicted field category (Random Forest, normalized):", label_encoder.inverse_transform(prediction_normalized_rf)[0])
print("Predicted field category (HistGradientBoostingClassifier, normalized):", label_encoder.inverse_transform(prediction_normalized_hgb)[0])