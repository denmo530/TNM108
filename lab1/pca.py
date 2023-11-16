from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
cancer = load_breast_cancer()

# Plot Histograms of malignent and benign classes 
fig, axes = plt.subplots(10, 3, figsize=(12, 9))
malignant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]
ax = axes.ravel()

for i in range (30): 
    _, bins = np.histogram(cancer.data[:, i], bins=40)
    ax[i].hist(malignant[:, i], bins=bins, color="r", alpha=0.5)
    ax[i].hist(benign[:,i],bins=bins,color='g',alpha=0.3) 
    ax[i].set_title(cancer.feature_names[i],fontsize=9)

    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].set_yticks(())

ax[0].legend(['malignant','benign'],loc='best',fontsize=8)
plt.tight_layout() 
plt.show()

# just convert the scikit learn data-set to pandas data-frame.
cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)
plt.subplot(1,2,1) # first plot
plt.scatter(cancer_df['worst symmetry'], cancer_df['worst texture'],
 s=cancer_df['worst area']*0.05, color='magenta', label='check',alpha=0.3)
plt.xlabel('Worst Symmetry',fontsize=12)
plt.ylabel('Worst Texture',fontsize=12)
plt.subplot(1,2,2) # 2nd plot
plt.scatter(cancer_df['mean radius'], cancer_df['mean concave points'],
 s=cancer_df['mean area']*0.05, color='purple',label='check', alpha=0.3)
plt.xlabel('Mean Radius',fontsize=12)
plt.ylabel('Mean Concave Points',fontsize=12)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() #instantiate
# compute the mean and standard which will be used in the next command
scaler.fit(cancer.data)
X_scaled=scaler.transform(cancer.data)
# we check the minimum and maximum of the scaled features which we expect to be 0 and 1
print("after scaling minimum", X_scaled.min(axis=0))

from sklearn.decomposition import PCA
pca=PCA(n_components=3)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)
print("shape of X_pca", X_pca.shape) # let's check the shape of X_pca array

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)

# Xax=X_pca[:,0]
# Yax=X_pca[:,1]
Xax=X_pca[:,0]
Yax=X_pca[:,2]
labels=cancer.target
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,label=labl[l],marker=marker[l],alpha=alpha[l])
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()

plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
plt.tight_layout()
plt.show()

feature_worst=list(cancer_df.columns[20:31]) # select the 'worst' features
import seaborn as sns
s=sns.heatmap(cancer_df[feature_worst].corr(),cmap='coolwarm')
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()
