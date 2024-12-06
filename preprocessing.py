import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()  # Remove missing values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data
