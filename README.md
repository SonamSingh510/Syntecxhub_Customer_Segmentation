# 🛍️ Customer Segmentation using K-Means

In the retail industry, understanding customer personas is essential for targeted marketing and customer retention. Unlike supervised learning, this project uses **Clustering** to find hidden patterns in data without pre-defined labels. Using the **K-Means** algorithm, I identified natural groupings within a mall customer dataset to derive data-driven business insights.

## 🛠️ Key Features

* **Data Preprocessing & Scaling:** Implemented `StandardScaler` to normalize features (Annual Income and Spending Score), ensuring the distance-based K-Means algorithm performs accurately.
* **The Elbow Method:** Conducted an iterative WCSS (Within-Cluster Sum of Squares) analysis to mathematically determine the optimal number of clusters ($k$).
* **K-Means Clustering:** Applied the optimized algorithm to segment the customer base into 5 distinct categories.
* **Cluster Visualization:** Generated high-quality 2D scatter plots highlighting the segments and their respective **Centroids**.
* **Segment Profiling:** Analyzed the characteristics of each group (Age, Income, Spending habits) to suggest specific marketing actions for a business report.

## 📊 Customer Profiles Identified

1. **Target:** High Income, High Spending (Primary focus for premium rewards).
2. **Sensible:** High Income, Low Spending (Target for high-value promotions).
3. **Standard:** Average Income, Average Spending (Steady customer base).
4. **Careless:** Low Income, High Spending (Impulsive buyers).
5. **Frugal:** Low Income, Low Spending (Budget-conscious).

## 💻 Tech Stack

* **Language:** Python 3.10
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Model Persistence:** `joblib`

## 📂 Project Structure

```text
├── custseg.py                   # Main clustering & visualization logic
├── Mall_Customers.csv           # Input dataset
├── customer_segments_output.csv # Final report with assigned cluster labels
└── customer_segmentation.pkl    # Serialized K-Means model
```

## ⚙️ How to Run

1. **Clone the repo:** `git clone https://github.com/SonamSingh510/customer-segmentation.git`
2. **Install dependencies:** `pip install pandas scikit-learn matplotlib seaborn`
3. **Execute:** `python customer_segmentation.py`
