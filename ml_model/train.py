import pandas as pd
from sklearn.cluster  import KMeans
import joblib
def train_model():
    # Load dataset
    data =pd.read_csv("E:\my_ml_project\data\Mall_Customers.csv")
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
   
       # Based on the Elbow Method, let's choose 5 clusters
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    kmeans.fit(X)  # Train the KMeans model
    # Add the clusters to the dataset
    clusters = kmeans.labels_
    data['cluster'] = clusters    # Initialize and train the model

    joblib.dump(kmeans, 'model.pkl')
      # Return the trained model
    return kmeans
if __name__ == '__main__':
    train_model()
