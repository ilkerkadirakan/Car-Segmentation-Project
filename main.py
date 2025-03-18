import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

# Step 1: Definition of the Problem
# Problem: Grouping vehicles based on their specifications and pricing to identify customer segments.

# Step 2: Examination of the Data
print("\nExamining the Data...")
data = pd.read_excel('2.xlsx')
print(data.info())

# Separate numerical and categorical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

selected_columns_show=['sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']


# Plot initial data distribution
plt.figure(figsize=(12, 6))
sns.histplot(data[selected_columns_show], kde=True, palette="viridis")
plt.title("Initial Data Distribution")
plt.savefig('graphs\step2_initial_data_distribution.png')
plt.show()

# Number of numerical and categorical columns
numerical_count = len(numerical_columns)
categorical_count = len(categorical_columns)
total_count = numerical_count + categorical_count

# Labels and sizes for the pie chart
labels = ['Numerical Columns', 'Categorical Columns']
sizes = [numerical_count, categorical_count]
colors = ['#66b3ff', '#99ff99']

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Distribution of Numerical and Categorical Columns')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.savefig('graphs\step2_column_distribution.png')
plt.show()


# Step 3: Preparation of the Data for Analysis
print("\nPreparing the Data...")
# Combine Encoding for Categorical Variables
if 'manufact' in categorical_columns and 'model' in categorical_columns:
    data['manufact_model'] = data['manufact'] + '_' + data['model']
    label_encoder = LabelEncoder()
    data['manufact_model_Encoded'] = label_encoder.fit_transform(data['manufact_model'])
    categorical_columns.append('manufact_model_Encoded')
    categorical_columns.remove('manufact')
    categorical_columns.remove('model')

data_original = data.copy()

# Fill missing values
for col in numerical_columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

for col in categorical_columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])


# Identify rows affected by IQR clipping or missing value filling
affected_rows = (data_original != data).any(axis=1)

# Save prepared data and highlight affected rows in Excel
output_file = 'sheets\step3_filled_data_highlighted.xlsx'
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

# Save the data to the Excel file
data.to_excel(writer, sheet_name='Prepared Data', index=False)

# Get the workbook and worksheet objects
workbook = writer.book
worksheet = writer.sheets['Prepared Data']

# Define the green background format
highlight_format = workbook.add_format({'bg_color': '#00FF00'})

# Highlight rows where any column was modified
for row_idx, modified in enumerate(affected_rows, start=1):  # Excel rows start at 1
    if modified:
        worksheet.set_row(row_idx, None, highlight_format)

# Save and close the workbook
writer.close()



#New Data Copy for iqr
data_original = data.copy()

# Handle Outliers using IQR
Q1 = data[numerical_columns].quantile(0.25)
Q3 = data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

for col in numerical_columns:
    lower_bound = Q1[col] - 1.5 * IQR[col]
    upper_bound = Q3[col] + 1.5 * IQR[col]
    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)



selected_columns_show=['sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']

# Plot before and after IQR clipping as side-by-side box plots
plt.figure(figsize=(18, 6))
sns.boxplot(data=data_original[selected_columns_show], palette="coolwarm")
plt.title("Data Distribution before Cleaning")
plt.savefig('graphs\step3_before_cleaning_data_distribution.png')
plt.show()



# Plot data distribution after cleaning
plt.figure(figsize=(18, 6))
sns.boxplot(data=data[selected_columns_show], palette="coolwarm")
plt.title("Data Distribution After Cleaning")
plt.savefig('graphs\step3_cleaned_data_distribution.png')
plt.show()

# Identify rows affected by IQR clipping or missing value filling
affected_rows = (data_original != data).any(axis=1)

# Save prepared data and highlight affected rows in Excel
output_file = 'sheets\step3_prepared_data_highlighted.xlsx'
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

# Save the data to the Excel file
data.to_excel(writer, sheet_name='Prepared Data', index=False)

# Get the workbook and worksheet objects
workbook = writer.book
worksheet = writer.sheets['Prepared Data']

# Define the yellow background format
highlight_format = workbook.add_format({'bg_color': '#FFEB9C'})

# Highlight rows where any column was modified
for row_idx, modified in enumerate(affected_rows, start=1):  # Excel rows start at 1
    if modified:
        worksheet.set_row(row_idx, None, highlight_format)

# Save and close the workbook
writer.close()

# Standardize numerical data
print("\nStandardizing Data...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[numerical_columns])


# Plot standardized data
plt.figure(figsize=(12, 6))
sns.histplot(X_scaled, kde=True, palette="muted")
plt.title("Standardized Data Distribution")
plt.savefig('graphs\step5_standardized_data_distribution.png')
plt.show()


# Step 4: Explanation of the Analysis Method

# Reduce dimensionality using PCA for visualization and performance improvement

# Apply PCA
print("\nApplying PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

# Plot PCA components
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, cmap='viridis')
plt.title("PCA Components")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('graphs\step5_pca_components.png')
plt.show()

# Step 5: Applying Clustering Algorithms


print("\nTuning K-Means Parameters with Grid Search...")

param_grid_kmeans = {
    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20],
    'max_iter': [300, 500],
    'algorithm': ['lloyd', 'elkan']
}
best_score_kmeans = -1
best_params_kmeans = None

for params in ParameterGrid(param_grid_kmeans):
    kmeans_temp = KMeans(
        n_clusters=params['n_clusters'],
        init=params['init'],
        n_init=params['n_init'],
        max_iter=params['max_iter'],
        algorithm=params['algorithm'],
        random_state=42
    )
    labels_temp = kmeans_temp.fit_predict(X_pca)
    sil_temp = silhouette_score(X_pca, labels_temp)
    if sil_temp > best_score_kmeans:
        best_score_kmeans = sil_temp
        best_params_kmeans = params

print(f"Best Silhouette Score (K-Means) from tuning: {best_score_kmeans:.4f}")
print("Best Parameters (K-Means):", best_params_kmeans)

print("\nPerforming K-Means Clustering...")
kmeans = KMeans(
    n_clusters=best_params_kmeans['n_clusters'],
    init=best_params_kmeans['init'],
    n_init=best_params_kmeans['n_init'],
    max_iter=best_params_kmeans['max_iter'],
    algorithm=best_params_kmeans['algorithm'],
    random_state=42
)
data['Cluster_KMeans'] = kmeans.fit_predict(X_pca)

    # Plot K-Means clustering results
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster_KMeans'], cmap='viridis', alpha=0.7)
plt.title("K-Means Clustering Results")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('graphs\step5_kmeans_clustering.png')
plt.show()

# Save K-Means clustering results to Excel
data.to_excel('sheets\step5_kmeans_results.xlsx', index=False)


print("\nTuning Agglomerative Clustering Parameters with Grid Search...")

param_grid_agg = {
    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'metric': ['euclidean', 'manhattan', 'cosine'],
    'linkage': ['ward', 'complete', 'average', 'single']
}

best_score_agg = -1
best_params_agg = None

for params in ParameterGrid(param_grid_agg):
    # Ward linkage + Euclidean dışı kombinasyonları atla (geçersiz)
    if params['linkage'] == 'ward' and params['metric'] != 'euclidean':
        continue

    agg_temp = AgglomerativeClustering(
        n_clusters=params['n_clusters'],
        metric=params['metric'],
        linkage=params['linkage']
    )
    labels_temp = agg_temp.fit_predict(X_pca)
    sil_temp = silhouette_score(X_pca, labels_temp)

    if sil_temp > best_score_agg:
        best_score_agg = sil_temp
        best_params_agg = params

print(f"Best Silhouette Score (Agglomerative) from tuning: {best_score_agg:.4f}")
print("Best Parameters (Agglomerative):", best_params_agg)

#############################################################
# -- Orijinal kod: Agglomerative Clustering (en iyi parametrelerle)
#############################################################
print("\nPerforming Agglomerative Clustering...")
agglomerative = AgglomerativeClustering(
    n_clusters=best_params_agg['n_clusters'],
    metric=best_params_agg['metric'],
    linkage=best_params_agg['linkage']
)
data['Cluster_Agglomerative'] = agglomerative.fit_predict(X_pca)

# Plot Agglomerative clustering results
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster_Agglomerative'], cmap='plasma', alpha=0.7)
plt.title("Agglomerative Clustering Results")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig('graphs\step5_agglomerative_clustering.png')
plt.show()

# Save Agglomerative clustering results to Excel
data.to_excel('sheets\step5_agglomerative_results.xlsx', index=False)

# Step 6: Presentation of the Analysis Results
# Evaluate Clustering with Silhouette Scores
print("\nEvaluating Clustering...")
silhouette_kmeans = silhouette_score(X_pca, data['Cluster_KMeans'])
silhouette_agglomerative = silhouette_score(X_pca, data['Cluster_Agglomerative'])
print(f"Silhouette Score (K-Means with PCA): {silhouette_kmeans}")
print(f"Silhouette Score (Agglomerative with PCA): {silhouette_agglomerative}")

# Visualize PCA-transformed Clusters
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster_KMeans'], cmap='viridis')
plt.title("K-Means Clustering with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Cluster_Agglomerative'], cmap='plasma')
plt.title("Agglomerative Clustering with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig('graphs\step6_clustering_visualization.png')
plt.show()

# Save final data with clustering results to Excel
data.to_excel('sheets\step6_final_results.xlsx', index=False)

# Step 7: Evaluation of the Results
# Critical assessment and future recommendations
print("\nEvaluation and Recommendations:")
print(f"K-Means Silhouette Score: {silhouette_kmeans:.2f}")
print(f"Agglomerative Silhouette Score: {silhouette_agglomerative:.2f}")
print("K-Means provided clear separations among clusters, but Agglomerative Clustering showed potential for more nuanced groupings.")


from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import numpy as np


def dunn_index(data, labels):
    unique_clusters = np.unique(labels)
    num_clusters = len(unique_clusters)

    if num_clusters < 2:
        return np.nan  # Dunn index cannot be computed for a single cluster

    centroids = np.array([data[labels == cluster].mean(axis=0) for cluster in unique_clusters])
    intra_cluster_distances = [cdist(data[labels == cluster], [centroids[cluster]]) for cluster in unique_clusters]
    max_intra_distance = np.max([np.max(distances) for distances in intra_cluster_distances])

    inter_cluster_distances = cdist(centroids, centroids)
    np.fill_diagonal(inter_cluster_distances, np.inf)  # Ignore diagonal
    min_inter_distance = np.min(inter_cluster_distances)

    return min_inter_distance / max_intra_distance


# Calculate additional cluster validity metrics
calinski_harabasz_kmeans = calinski_harabasz_score(X_pca, data['Cluster_KMeans'])
calinski_harabasz_agg = calinski_harabasz_score(X_pca, data['Cluster_Agglomerative'])

davies_bouldin_kmeans = davies_bouldin_score(X_pca, data['Cluster_KMeans'])
davies_bouldin_agg = davies_bouldin_score(X_pca, data['Cluster_Agglomerative'])

dunn_kmeans = dunn_index(X_pca, data['Cluster_KMeans'].to_numpy())
dunn_agg = dunn_index(X_pca, data['Cluster_Agglomerative'].to_numpy())

# Display the results
print("\nCluster Validity Metrics:")
print(f"Calinski-Harabasz Index (K-Means): {calinski_harabasz_kmeans:.2f}")
print(f"Calinski-Harabasz Index (Agglomerative): {calinski_harabasz_agg:.2f}")
print(f"Davies-Bouldin Index (K-Means): {davies_bouldin_kmeans:.2f}")
print(f"Davies-Bouldin Index (Agglomerative): {davies_bouldin_agg:.2f}")
print(f"Dunn Index (K-Means): {dunn_kmeans:.2f}")
print(f"Dunn Index (Agglomerative): {dunn_agg:.2f}")

def colorize_clusters_in_excel_with_full_row(df, output_file, cluster_columns):
    """
    Excel dosyasına iki ayrı sayfa halinde küme değerlerine göre tüm satırı renklendiren fonksiyon.

    Args:
        df (pd.DataFrame): Renklendirilecek veri çerçevesi.
        output_file (str): Kaydedilecek Excel dosyasının adı.
        cluster_columns (list): Renklendirilecek küme kolonlarının isimleri.
    """
    # Excel Writer oluştur
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    # Renk haritası
    color_map = {
        0: '#FFC7CE',  # Pembe
        1: '#C6EFCE',  # Açık yeşil
        2: '#FFEB9C',  # Sarı
        3: '#BDD7EE',  # Açık mavi
        4: '#FCE4D6',  # Turuncu
        5: '#E2EFDA',  # Yeşil ton
    }

    # Her küme kolonunu ayrı sayfada işlemek için döngü
    for col_name in cluster_columns:
        if col_name not in df.columns:
            print(f"Uyarı: '{col_name}' kolon bulunamadı, atlanıyor.")
            continue

        # Yeni sayfa adı
        sheet_name = f"Clusters_{col_name}"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        col_idx = df.columns.get_loc(col_name)  # Küme kolonunun indeksi

        # Her satırı renklendir
        for row_idx, cluster_val in enumerate(df[col_name], start=1):  # 1. satırdan başla
            if cluster_val in color_map:
                bg_color = color_map[cluster_val]
                fmt = workbook.add_format({'bg_color': bg_color})

                # Tüm satırı renklendir
                worksheet.set_row(row_idx, None, fmt)

    writer.close()
    print(f"{output_file} başarıyla kaydedildi. İki ayrı sayfa oluşturuldu.")


colorize_clusters_in_excel_with_full_row(
    df=data,
    output_file='sheets\step6_final_results_colored.xlsx',
    cluster_columns=['Cluster_KMeans', 'Cluster_Agglomerative']
)



from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr

def cluster_validity_correlation(X, labels, metric='euclidean'):
    """
    Veri seti (X) ve kümeler (labels) verildiğinde:
    1) Proximity Matrix (distance matrix) oluşturur.
    2) Ideal Similarity Matrix oluşturur. (Aynı kümedekiler = 1, farklı kümedekiler = 0)
    3) Bu iki matrisin üst üçgenini düzleştirerek Pearson korelasyonunu hesaplar.
    
    Parametreler:
        X       : Veri seti, shape = (n_samples, n_features)
        labels  : Küme etiketleri, shape = (n_samples,)
        metric  : Uzaklık ölçümü ("euclidean", "cosine", "manhattan", vb.)
    Döndürür:
        corr    : Pearson korelasyon katsayısı
        p_value : İlgili p-değeri
    """
    # 1) Proximity Matrix (Distance Matrix)
    distance_matrix = pairwise_distances(X, metric=metric)  # NxN boyutlu

    n = len(labels)

    # 2) Ideal Similarity Matrix
    #    Aynı kümede iseler 1, değilseler 0.
    ideal_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                ideal_matrix[i, j] = 1

    # 3) Matrislerin Üst Üçgenini (diagonal üstü) düzleştirerek vektör oluşturma
    triu_indices = np.triu_indices(n, k=1)   # k=1 => diagonal hariç üst üçgen
    distance_vector = distance_matrix[triu_indices]
    ideal_vector = ideal_matrix[triu_indices]

    # 4) Korelasyon Hesaplama (Pearson)
    corr, p_value = pearsonr(distance_vector, ideal_vector)

    return corr, p_value


# Örnek kullanım:
if __name__ == "_main_":
    # Diyelim ki 5 gözlem ve 2 boyutlu veri olsun
    X_example = np.array([
        [1.0, 2.0],
        [1.2, 1.8],
        [5.0, 8.0],
        [6.0, 8.1],
        [5.5, 7.9]
    ])
    
    # Küme etiketleri
    labels_example = np.array([0, 0, 1, 1, 1])

    # Korelasyonu hesapla
    corr_score, p_val = cluster_validity_correlation(
        X_example,
        labels_example,
        metric='euclidean'
    )

    print(f"Correlation Score: {corr_score:.4f}")
    print(f"P-Value           : {p_val:.4f}")
