Customer Segmentation with Cross-Validation
📌 Project Overview

This project applies KMeans clustering to segment customers based on their Age, Annual Income, and Spending Score.
It integrates K-Fold Cross Validation to evaluate clustering quality, allows user input for new customer data, and visualizes clusters, centroids, and user points.

The goal is to help businesses understand customer groups and make data-driven decisions for marketing, targeting, and strategy.

⚙️ Features

Data Preprocessing: Standard scaling of features.

Cross Validation: Evaluate clustering quality with Silhouette Score.

Clustering: KMeans algorithm with configurable number of clusters.

User Input: Enter new customer data interactively (Age, Income, Spending Score).

Cluster Assignment: Predicts cluster for user entries and computes distance to centroid.

Visualization:

Customer dataset clusters.

Centroids of each cluster.

User-provided data points annotated with cluster & distance.

📂 Project Structure
Customer_Segmentation_CV/
│-- customer_data.csv          # Dataset (Age, Income, Spending Score)
│-- Customer_Segmentation.py   # Main script
│-- README.md                  # Project documentation

🚀 How to Run
1. Install dependencies
pip install numpy pandas matplotlib scikit-learn

2. Run the script
python Customer_Segmentation.py

3. Provide user input

When prompted, enter rows in the format:

Age, Annual_Income, Spending_Score


Example:

34, 45000, 60
28, 65000, 40


Press Enter on an empty line to finish input.

📊 Output

Cross Validation Results:

Average Silhouette Score (with standard deviation).

Cluster Prediction for User Data:

Predicted cluster label.

Distance to centroid.

Visualization:

Scatter plot of customers (Income vs Spending Score).

Centroids shown as black "X".

User points marked with diamond symbols and annotated.

🧠 Example Console Output
=== Cross Validation Results ===
Average Silhouette Score: 0.5213 (+/- 0.0421)

Results for entered customers:
 Age  Annual_Income  Spending_Score  Predicted_Cluster  Distance_to_Centroid
  34         45000              60                  2                  1.23
  28         65000              40                  1                  2.05

📈 Example Visualization

Customers grouped by clusters.

Centroids (black X).

User-entered points (diamond markers) labeled with cluster & distance.

🔮 Future Improvements

Support automatic cluster selection using Elbow Method / Silhouette Analysis.

Add different clustering algorithms (DBSCAN, Agglomerative).

Deploy as a Flask/Django web app for interactive usage.

👨‍💻 Author

Developed by Anas Mohamed ✨
For learning Clustering, Cross Validation, and Data Visualization with Python.
