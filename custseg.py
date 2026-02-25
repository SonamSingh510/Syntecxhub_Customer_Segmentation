import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load Dataset
# Typical 'Mall_Customers.csv' has: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
try:
    df = pd.read_csv('Mall_Customers.csv')
    print("--- Dataset Loaded ---")
    print(df.head())
except FileNotFoundError:
    print("Error: Mall_Customers.csv not found. Please ensure it's in your project folder.")

# 2. Preprocessing & Scaling
# We usually cluster based on 'Annual Income' and 'Spending Score'
X = df.iloc[:, [3, 4]].values 

# Scaling is CRITICAL for K-Means (Distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. The Elbow Method (Choose k)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

# 4. Applying K-Means (k=5 is usually the 'elbow' for this dataset)
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = y_kmeans

# 5. Visualizing Clusters
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(k):
    plt.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1], 
                s=100, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids', edgecolors='black')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend()
plt.show()

# 6. Profiling Clusters (Derive Marketing Actions)
print("\n--- Cluster Profiles (Means) ---")
profile = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(profile)

# 7. Save Model & Labels
joblib.dump(kmeans, 'customer_segmentation_model.pkl')
df.to_csv('customer_segments_output.csv', index=False)
print("\n✅ Success! Segmented report saved as 'customer_segments_output.csv'")