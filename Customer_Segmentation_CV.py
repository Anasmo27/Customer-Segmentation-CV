"""
Customer Segmentation with Cross-Validation
- Train KMeans on a dataset (Age, Annual_Income, Spending_Score)
- Use KFold Cross Validation to evaluate clustering quality (Silhouette Score)
- Allow user to input rows -> predict cluster, compute distance to centroid
- Visualize clusters, centroids, and user points
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import math

# -------------------------
# 1) Load dataset
# -------------------------
file_path = r"C:\Users\ASUS\Desktop\python test\Customer_Segmentation_CV\customer_data.csv"
data = pd.read_csv(file_path)

required_cols = ["Age", "Annual_Income", "Spending_Score"]
if not all(col in data.columns for col in required_cols):
    raise ValueError(f"CSV file must contain columns: {required_cols}")

X = data[required_cols].values

# -------------------------
# 2) Preprocessing (scaling)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 3) Cross Validation (KFold)
# -------------------------
N_CLUSTERS = 4
kf = KFold(n_splits=5, shuffle=True, random_state=42)
sil_scores = []

for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X_train)
    preds = kmeans.predict(X_test)

    if len(np.unique(preds)) > 1:
        sil = silhouette_score(X_test, preds)
        sil_scores.append(sil)

print("\n=== Cross Validation Results ===")
print(f"Average Silhouette Score: {np.mean(sil_scores):.4f} (+/- {np.std(sil_scores):.4f})")

# -------------------------
# 4) Train final model on full dataset
# -------------------------
final_model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
final_model.fit(X_scaled)
centroids = final_model.cluster_centers_

# -------------------------
# Helper: parse user input
# -------------------------
def parse_user_rows():
    print("\nEnter customer rows (Age, Annual_Income, Spending_Score).")
    print("Example: 34, 45000, 60")
    print("Press Enter on empty line to finish.\n")

    rows = []
    while True:
        line = input("Row: ").strip()
        if line == "":
            break
        try:
            age, income, score = map(float, line.split(","))
            rows.append((age, income, score))
        except:
            print("❌ Invalid format. Please use: Age, Income, Score")
    return rows

# -------------------------
# 5) Get user rows
# -------------------------
user_rows = parse_user_rows()
if len(user_rows) == 0:
    print("No input provided — exiting.")
    exit()

user_df = pd.DataFrame(user_rows, columns=["Age", "Annual_Income", "Spending_Score"])
user_scaled = scaler.transform(user_df.values)

# -------------------------
# 6) Predict user clusters + distances
# -------------------------
user_preds = final_model.predict(user_scaled)

def euclidean(a, b):
    return math.sqrt(((a - b) ** 2).sum())

distances = [euclidean(user_scaled[i], centroids[user_preds[i]]) for i in range(len(user_preds))]

out_df = user_df.copy()
out_df["Predicted_Cluster"] = user_preds
out_df["Distance_to_Centroid"] = np.round(distances, 4)
print("\nResults for entered customers:")
print(out_df.to_string(index=False))

# -------------------------
# 7) Visualization
# -------------------------
labels = final_model.labels_
centroids_orig = scaler.inverse_transform(centroids)

fig, ax = plt.subplots(figsize=(10, 6))

# dataset points
scatter = ax.scatter(
    X[:, 1],  # Annual Income
    X[:, 2],  # Spending Score
    c=labels, cmap="tab10", alpha=0.6, edgecolor="k", s=50
)

# centroids
ax.scatter(
    centroids_orig[:, 1],
    centroids_orig[:, 2],
    marker="X", s=200, c="black", label="Centroids"
)

# user points
for i, row in user_df.iterrows():
    ax.scatter(row["Annual_Income"], row["Spending_Score"],
               marker="D", s=120, c=[user_preds[i]], cmap="tab10",
               edgecolor="white", label=f"User {i+1}" if i == 0 else None)
    ax.annotate(f"C{user_preds[i]}, d={distances[i]:.2f}",
                (row["Annual_Income"], row["Spending_Score"]),
                textcoords="offset points", xytext=(6,6), fontsize=9, weight="bold")

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segmentation with KMeans (Cross-Validation + User Inputs)")
ax.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nDone ✅ — rerun the script to test with new inputs.")
