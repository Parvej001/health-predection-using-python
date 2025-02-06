import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import statistics

# Paths to datasets
TRAINING_DATA_PATH = "dataset/Training.csv"  # Ensure this path is correct
TESTING_DATA_PATH = "dataset/Testing.csv"    # Ensure this path is correct

# Function to safely read CSV files
def read_csv_safe(filepath):
    try:
        # Attempt to read the file and drop empty columns
        data = pd.read_csv(filepath).dropna(axis=1)
        print(f"Successfully loaded file: {filepath}")
        return data
    except pd.errors.ParserError as e:
        print(f"ParserError while reading {filepath}: {e}")
        exit()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        exit()

# Reading training data
training_data = read_csv_safe(TRAINING_DATA_PATH)

# Visualizing the dataset balance
disease_counts = training_data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
plt.title("Disease Count Distribution")
plt.show()

# Encoding the target column using LabelEncoder
encoder = LabelEncoder()
training_data["prognosis"] = encoder.fit_transform(training_data["prognosis"])

# Splitting features and target
X = training_data.iloc[:, :-1]
y = training_data.iloc[:, -1]

# Splitting data into train-test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Cross-validation scoring function
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Initializing models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Evaluating models using cross-validation
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)
    print("==" * 30)
    print(f"{model_name}")
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Score: {np.mean(scores):.2f}")

# Training and evaluating the models
final_models = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{model_name} Accuracy on Test Data: {acc * 100:.2f}%")
    final_models[model_name] = model

    # Plot confusion matrix
    cf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True, fmt='d')
    plt.title(f"Confusion Matrix for {model_name} on Test Data")
    plt.show()

# Reading testing data
testing_data = read_csv_safe(TESTING_DATA_PATH)

# Preparing the testing data
test_X = testing_data.iloc[:, :-1]
test_Y = encoder.transform(testing_data.iloc[:, -1])

# Combining predictions from all models using mode
svm_preds = final_models["SVC"].predict(test_X)
nb_preds = final_models["Gaussian NB"].predict(test_X)
rf_preds = final_models["Random Forest"].predict(test_X)

# Final prediction by mode
final_preds = [statistics.mode([svm, nb, rf]) for svm, nb, rf in zip(svm_preds, nb_preds, rf_preds)]
final_acc = accuracy_score(test_Y, final_preds)
print(f"Accuracy of Combined Model on Test Data: {final_acc * 100:.2f}%")

# Plot confusion matrix for combined model
cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix for Combined Model on Test Data")
plt.show()

# Function for making predictions
symptom_index = {symptom: idx for idx, symptom in enumerate(X.columns)}

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(symptom_index)
    for symptom in symptoms:
        index = symptom_index.get(symptom, -1)
        if index != -1:
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = encoder.classes_[final_models["Random Forest"].predict(input_data)[0]]
    nb_prediction = encoder.classes_[final_models["Gaussian NB"].predict(input_data)[0]]
    svm_prediction = encoder.classes_[final_models["SVC"].predict(input_data)[0]]
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])

    return {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }

# Example usage
example_symptoms = "symptom_1,symptom_5,symptom_9"
print(predictDisease(example_symptoms))