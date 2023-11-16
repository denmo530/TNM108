from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

customer_data = pd.read_csv("./datasets/part2/shopping_data.csv")
data = customer_data.iloc[:, 3:5].values

Z = hierarchy.linkage(data, "ward")
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.show()
#Finding an interesting number of clusters in a dendrogram is the same as finding the largest horizontal space that doesn't have any vertical lines (the space with the longest vertical lines). This means that there's more separation between the clusters. = 5 clusters



cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
cluster.labels_

plt.scatter(x=data[:,0], y=data[:, 1], c=cluster.labels_, cmap='rainbow')
plt.xlabel("Annual income")
plt.ylabel("Spending score")
plt.show() 
