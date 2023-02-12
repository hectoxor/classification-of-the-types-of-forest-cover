
y = data['class']
X = data.drop(['class'], axis=1)

#Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
import numpy as np 
import matplotlib.pyplot as plt 

def plot_num_component_vs_explained_variance(X_train): 
    pca = PCA(n_components=X_train.shape[1])
    X_train_pca = pca.fit_transform(X_train)
    x_axis = [i+1 for i in range(X_train.shape[1])]
    y_axis = [np.sum(pca.explained_variance_ratio_[:i+1]) * 100 for i in range(X_train.shape[1])]
    print(y_axis)
    plt.clf() 
    plt.plot(x_axis, y_axis)
    plt.xlabel("Number of selected features")
    plt.ylabel("%age of explained variance")
    plt.title("Num Features vs Explained Variance")
    plt.show() 

plot_num_component_vs_explained_variance(X_train)
def make_pca(X_train, num_components): 
    pca = PCA(n_components = num_components)
    X_train_pca = pca.fit_transform(X_train)
    return pca, X_train_pca
  
pca_10, X_train_pca_10 = make_pca(X_train, num_components=10)
X_test_pca_10 = pca_10.transform(X_test)
print(sorted(Counter(y_train).items()))
import seaborn as sns

pc_df = pd.DataFrame(data=X_train_pca_10, columns = ['PC1', 'PC2','PC3','PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
pc_df['Cluster'] = y_train
sns.lmplot( x="PC9", y="PC10",
  data=pc_df, 
  fit_reg=False, 
  hue='Cluster', 
  legend=True,) 
cnt = sorted(Counter(y_train).items())
#rus = SMOTE(random_state=0, sampling_strategy={1:cnt[0][1], 2:cnt[1][1], 3:cnt[2][1], 4:12500, 5:20000, 6:23000, 7:cnt[6][1]}, k_neighbors = 5)
rus = SMOTE(random_state=0, sampling_strategy={1:169259, 2:226793, 3:100000, 4:100000, 5:80000, 6:80000, 7:90000}, k_neighbors = 10)
X_train_pca_10, y_train = rus.fit_resample(X_train_pca_10, y_train)
print(sorted(Counter(y_train).items()))
