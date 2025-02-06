import pandas as pd
import numpy as np

# Example symptoms and diseases
symptoms = [f"symptom_{i}" for i in range(1, 11)]  # 10 dummy symptoms
diseases = ["Disease_A", "Disease_B", "Disease_C"]  # 3 dummy diseases

# Create dummy training data
np.random.seed(42)
data = {
    **{symptom: np.random.randint(0, 2, 100) for symptom in symptoms},  # 100 rows for training
    "prognosis": np.random.choice(diseases, 100)
}

# Convert to DataFrame and save as Training.csv
df = pd.DataFrame(data)
df.to_csv("dataset/Training.csv", index=False)
print("Training.csv created!")
