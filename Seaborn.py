import seaborn as sns

pc_df = pd.DataFrame(data=X_train_pca_10, columns = ['PC1', 'PC2','PC3','PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
pc_df['Cluster'] = y_train
sns.lmplot( x="PC9", y="PC10",
  data=pc_df, 
  fit_reg=False, 
  hue='Cluster', 
  legend=True,) 
