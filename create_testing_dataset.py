import pandas as pd
import numpy as np

# Example symptoms and diseases (same as for Training)
symptoms = [f"symptom_{i}" for i in range(1, 11)]  # 10 dummy symptoms
diseases = ["Disease_A", "Disease_B", "Disease_C"]  # 3 dummy diseases

# Create dummy testing data
np.random.seed(42)
data = {
    **{symptom: np.random.randint(0, 2, 30) for symptom in symptoms},  # 30 rows for testing
    "prognosis": np.random.choice(diseases, 30)
}

# Convert to DataFrame and save as Testing.csv
df = pd.DataFrame(data)
df.to_csv("dataset/Testing.csv", index=False)
print("Testing.csv created!")
