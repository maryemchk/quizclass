import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Charger le DataFrame à partir du fichier CSV
df = pd.read_csv('student_data.csv')

# Préparation des données
X = df[['Attendue', 'Réponse', 'Temps', 'Difficulté']]  # Caractéristiques
y = df['Erreur']  # Cible

# Définition des colonnes pour le prétraitement
numeric_features = ['Attendue', 'Réponse', 'Temps']
categorical_features = ['Difficulté']

# Création du préprocesseur
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Création du pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])

# Entraînement du modèle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Évaluation du modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test du modèle sur un nouvel élève
new_data = {
    'Attendue': [14],
    'Réponse': [12],
    'Temps': [45],
    'Difficulté': ['Moyen']
}

new_df = pd.DataFrame(new_data)
new_X = new_df  # Aucun prétraitement nécessaire pour un seul échantillon
prediction = pipeline.predict(new_X)

if prediction[0] == 0:
    print("\nL'élève n'a pas de problème en mathématiques.")
else:
    print("\nL'élève a un problème en mathématiques.")
