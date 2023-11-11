# %% read data
import pandas as pd

df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
# 1 area A,
# 2 perimeter P,
# 3 compactness C = 4*pi*A/P^2,
# 4 length of kernel,
# 5 width of kernel,
# 6 asymmetry coefficient
# 7 length of kernel groove.
# 8 target
df.columns = [
    "area",
    "perimeter",
    "compactness",
    "length_kernel",
    "width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]
df.head()

# %%
import pandas as pd
df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
df.columns = ["area","perimeter","compactness","length_kernel","width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]

df.describe()


#%%
import seaborn as sns
import pandas as pd
df = pd.read_csv("seeds_dataset.txt", sep="\t+", header=None)
df.columns = ["area","perimeter","compactness","length_kernel","width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]


sns.scatterplot(
    x="area",
    y="asymmetry_coefficient",
    data=df,
    hue="target",
    legend="full",
)


# %% also try lmplot and pairplot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the seeds dataset (replace 'seeds_dataset.txt' with the actual file path)
column_names = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class"]
seeds = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, names=column_names)

# lmplot: Relationship between perimeter and compactness
sns.lmplot(x='perimeter', y='compactness', data=seeds, hue='class', fit_reg=False, height=6, aspect=1.5)
plt.title('Relationship between Perimeter and Compactness')
plt.show()

# pairplot: Pairwise relationships between variables
sns.pairplot(seeds, vars=["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove"], hue='class')
plt.suptitle('Pairwise Relationships in Seeds Dataset', y=1.02)
plt.show()

# %% determine the best numbmer of clusters
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score
import matplotlib.pyplot as plt

# Load the seeds dataset (replace 'seeds_dataset.txt' with the actual file path)
column_names = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class"]
seeds = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, names=column_names)

# Select relevant features
features = seeds[['perimeter', 'compactness']]

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Iterate over a range of cluster numbers and calculate homogeneity scores
cluster_range = range(2, 11)
homogeneity_scores = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_standardized)
    
    homogeneity_scores.append(homogeneity_score(seeds['class'], kmeans.labels_))

# Plot homogeneity scores
plt.plot(cluster_range, homogeneity_scores, marker='o')
plt.title('Homogeneity Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Homogeneity Score')
plt.show()


# use kmeans to loop over candidate number of clusters 
# store inertia and homogeneity score in each iteration


# %% 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import homogeneity_score
import matplotlib.pyplot as plt

# Load the seeds dataset (replace 'seeds_dataset.txt' with the actual file path)
column_names = ["area", "perimeter", "compactness", "length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_kernel_groove", "class"]
seeds = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, names=column_names)

# Select relevant features
features = seeds[['perimeter', 'compactness']]

# Standardize the features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Loop over a range of candidate numbers of clusters
cluster_range = range(2, 11)
inertia_values = []
homogeneity_scores = []

for n_clusters in cluster_range:
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_standardized)
    
    # Store inertia and homogeneity score
    inertia_values.append(kmeans.inertia_)
    homogeneity_scores.append(homogeneity_score(seeds['class'], kmeans.labels_))

# Plot the results
plt.figure(figsize=(12, 4))

# Plot inertia values
plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Inertia for Different Numbers of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

# Plot homogeneity scores
plt.subplot(1, 2, 2)
plt.plot(cluster_range, homogeneity_scores, marker='o')
plt.title('Homogeneity Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Homogeneity Score')

plt.tight_layout()
plt.show()
# ax = sns.lineplot(
#     x=list(inertia.keys()),
#     y=list(inertia.values()),
#     color="blue",
#     label="inertia",
#     legend=None,
# )
# ax.set_ylabel("inertia")
# ax.twinx()
# ax = sns.lineplot(
#     x=list(homogeneity.keys()),
#     y=list(homogeneity.values()),
#     color="red",
#     label="homogeneity",
#     legend=None,
# )
# ax.set_ylabel("homogeneity")
# ax.figure.legend()


# %%
