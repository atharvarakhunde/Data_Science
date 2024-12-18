import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

file_path = "Titanic-Dataset.csv"
titanic_data = pd.read_csv(file_path)

titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
titanic_data = titanic_data.drop(columns=['Cabin'])

#SibSp=sibling .....Parch=parents/children
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']

titanic_data['Title'] = titanic_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
title_encoder = LabelEncoder()
titanic_data['Title'] = title_encoder.fit_transform(titanic_data['Title'])

titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

titanic_data = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch'])

X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(report)

cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
cross_val_scores = []

for train_idx, test_idx in cv.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
    
    best_model.fit(X_train_cv, y_train_cv)
    cv_preds = best_model.predict(X_test_cv)
    cv_score = accuracy_score(y_test_cv, cv_preds)
    cross_val_scores.append(cv_score)

print(f"Cross-validation accuracy: {sum(cross_val_scores) / len(cross_val_scores) * 100:.2f}%")
