# Vehicle Customer Segmentation

## Project Overview

This project aims to segment vehicle customers based on various car features and pricing. By identifying meaningful clusters, insights into customer preferences and market trends will be provided.

## Team Members

- **Eren Ergin**
- **Enes Gaşi**
- **Kadir İlker Akan**
- **Muhammed Yusuf Lale**


## Problem Statement

The goal of this project is to group different car models based on features such as engine size, horsepower, dimensions, price, and fuel efficiency. The results can help manufacturers optimize their pricing strategies and product portfolios.

## Dataset Description

- **Dataset Size**: 157 rows, 26 columns
- **Numerical Columns (20)**: Includes features like sales, resale value, price, engine size, horsepower, vehicle dimensions, weight, fuel capacity, and miles per gallon.
- **Categorical Columns (6)**: Includes information like manufacturer, model, and vehicle type.

### Missing Data Handling

- **Resale and zresale columns**: Contained 22.93% missing data.
- **Other minor missing values**: Found in price, engine size, horsepower, wheelbase, width, length, weight, fuel capacity, and MPG columns.
- **Missing Data Imputation**:
  - Median was used for numerical columns.
  - Mode was used for categorical columns.

## Data Preprocessing

### Encoding

- Categorical columns were converted to numerical values to be suitable for machine learning algorithms.
- Manufacturer and model columns were combined to create a more meaningful feature.

### Outlier Handling

- Outliers were detected and capped using the **Interquartile Range (IQR)** method.
- The goal was to keep the data balanced without losing significant information.

### Standardization

- Numerical columns were scaled to the [0,1] range using **MinMaxScaler**.
- Standardization improved the efficiency of clustering algorithms.

## Analysis Method
### Principal Component Analysis (PCA)

- **Why PCA?**
  - Provides visualization by reducing high-dimensional datasets to 2 dimensions.
  - Helps understand data structure by preserving variance.
  - Enhances the performance of clustering algorithms.
- **PCA Steps:**
  1. Compute the covariance matrix.
  2. Calculate eigenvalues and eigenvectors.
  3. Project onto the 2 components with the largest eigenvalues.

### Clustering Algorithms

#### 1. K-Means Clustering

- **Objective**: Partition the data into `k` clusters.
- **Optimization Parameters**:
  - Number of clusters: 2 to 10
  - Initialization methods: `k-means++`, `random`
  - Optimization algorithm: `lloyd`, `elkan`
  - Best parameters were determined using grid search.

#### 2. Agglomerative Clustering

- **Approach**: Hierarchically groups data based on cluster similarities.
- **Optimization Parameters**:
  - Number of clusters: 2 to 10
  - Distance metrics: `euclidean`, `manhattan`, `cosine`
  - Linkage methods: `ward`, `complete`, `average`, `single`
- **Best Parameters**:
  - K-Means: `n_clusters=3`, `init='k-means++'`, `max_iter=10`
  - Agglomerative: `n_clusters=3`, `metric='cosine'`, `linkage='complete'`

## Results & Evaluation

### Clustering Performance Metrics

| Metric                        | K-Means | Agglomerative |
| ----------------------------- | ------- | ------------- |
| **Silhouette Score**          | 0.5220  | 0.5203        |
| **Calinski-Harabasz Index**   | 217.47  | 213.17        |
| **Davies-Bouldin Index**      | 0.65    | 0.65          |
| **Dunn Index**                | 0.92    | 0.90          |

### Key Insights

- K-Means clustering performed slightly better than Agglomerative Clustering.
- Market segments identified:
  - **Economy Class** (affordable, fuel-efficient cars)
  - **Performance Vehicles** (high horsepower, sports cars)
  - **Luxury Vehicles** (high price, large dimensions)
- **Strategic Uses**:
  - Manufacturers can tailor marketing strategies to these segments.
  - Product development processes can be optimized.

### Cluster Validity Check

- **Correlation between Proximity Matrix and Ideal Similarity Matrix**: **-0.6711**
  - Negative correlation confirms strong cluster separation.

## File Structure

```
Car-Segmentation-Project/
├── graphs/
│   ├── step2_column_distribution.png
│   ├── step2_initial_data_distribution.png
│   ├── step3_before_cleaning_data_distribution.png
│   ├── step3_cleaned_data_distribution.png
│   ├── step5_agglomerative_clustering.png
│   ├── step5_kmeans_clustering.png
│   ├── step5_pca_components.png
│   ├── step5_standardized_data_distribution.png
│   ├── step6_clustering_visualization.png
│
├── sheets/
│   ├── step3_filled_data_highlighted.xlsx
│   ├── step3_prepared_data_highlighted.xlsx
│   ├── step5_agglomerative_results.xlsx
│   ├── step5_kmeans_results.xlsx
│   ├── step6_final_results.xlsx
│   ├── step6_final_results_colored.xlsx
│   ├── 2.xlsx
│
├── main.py
├── README.md  # This document
├── requirements.txt
├── streamlit_app.py
```

## Conclusion

- **K-Means was selected as the best clustering method.**
- Clustering analysis provided significant insights into market segmentation.
- Results can be used in **marketing, pricing optimization, and target audience identification**.

## Running the Code

### Requirements

Install dependencies with the following command:

```bash
pip install -r requirements.txt
```

### Execution

Run the Python script:

```bash
python main.py
```

All output files and visualizations will be generated in the respective folders.

**We welcome feedback! 🚀**

