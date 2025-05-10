import pandas as pd
from neo4j import GraphDatabase

# Initialize the Neo4j driver
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "mariaadnan"))

# Cypher query to retrieve features and target variable
query = """
MATCH (j:Journal)
RETURN j.Experience AS Experience, 
       j.DepressiveDisorder AS DepressiveDisorder, 
       j.MentalDisorders AS MentalDisorders, 
       j.Community AS Community, 
       j.Attitude AS Attitude, 
       j.MentalAssociation AS MentalAssociation, 
       j.Patients AS Patients, 
       j.Adolescent AS Adolescent,
       j.medicalCitationCount AS medicalCitationCount,
       j.is_medicine AS is_medicine
"""

# Run the query and store the results in a DataFrame
with driver.session() as session:
    result = session.run(query)
    df = pd.DataFrame([record.data() for record in result])

# Check the first few rows of the DataFrame
print(df.head())


# Define features (X) and target (y)
X = df[['Experience', 'DepressiveDisorder', 'MentalDisorders', 'Community', 
        'Attitude', 'MentalAssociation', 'Patients', 'Adolescent', 'medicalCitationCount']]
y = df['is_medicine']  # Target variable: is_medicine


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the resulting datasets
print(X_train.shape, X_test.shape)


from sklearn.ensemble import RandomForestClassifier

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed classification report
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# Example hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# New data for prediction (new journal's features)
new_data = pd.DataFrame({
    'Experience': [1],
    'DepressiveDisorder': [0],
    'MentalDisorders': [1],
    'Community': [0],
    'Attitude': [0],
    'MentalAssociation': [1],
    'Patients': [0],
    'Adolescent': [1],
    'medicalCitationCount': [120]
})

# Make predictions
predictions = clf.predict(new_data)
print("Prediction for new data:", predictions)
