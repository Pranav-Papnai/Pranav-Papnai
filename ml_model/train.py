import pandas as pd
from sklearn.cluster  import KMeans
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
def train_model():
    # Load dataset
    data =pd.read_csv(r'E:\my_ml_project\ml_model\Mall_Customers.csv')
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Based on the Elbow Method, let's choose 5 clusters
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)  # Train the KMeans model
    # Add the clusters to the dataset
    clusters = kmeans.labels_
    data['cluster'] = clusters    # Initialize and train the model

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', palette='Paired')
    plt.title('Customer segments')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()

    joblib.dump(kmeans, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
  # Return the trained model
    return kmeans
if __name__ == '__main__':
    train_model()
