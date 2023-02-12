from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from google.colab import drive
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
drive.mount('/content/drive')
%cd /content/drive/MyDrive
data = pd.read_csv('covertype_train.csv')

!pip install optuna
import optuna

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


def nn_objective(trial):

    

    #hidden_layer_spec = (15,15,15)

    # Uncomment the following section if you also want Optuna to try different
    # architectures. In this snippet, Optuna selects the number of layers and number
    # of neurons per layer
    layers = []
    n_layers = trial.suggest_int("n_layers", 3, 4)
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"n_units_l{i}", 1, 50))
    hidden_layer_spec = tuple(layers)
    
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-1)

    model = MLPClassifier(
                hidden_layer_sizes = hidden_layer_spec, 
                activation = 'relu',
                solver = 'adam',
                alpha = alpha,
                learning_rate_init = learning_rate_init,
                tol = 0.0001,
                n_iter_no_change= 10,
                max_iter = 10000
            )
    
    model.fit(X_train_pca_10, y_train)

    # Evaluate the accuracy on the test set
    y_pred = model.predict(X_test_pca_10)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(nn_objective, n_trials=50)

# Print the best hyperparameters
print("Best NN hyperparameters: ", study.best_params)



nn_clf_pca_10 = MLPClassifier(
    hidden_layer_sizes = (40, 30, 20, 10, 5),
    activation = 'relu',
    solver = 'adam',
    learning_rate_init = 0.001,
    tol = 0.001,
    n_iter_no_change=10,
    max_iter = 10000
)


import time 

pca_10_train_start_time = time.time() 
nn_clf_pca_10.fit(X_train_pca_10, y_train)
pca_10_train_total_time = time.time() - pca_10_train_start_time

pca_10_test_start_time = time.time()
pca_10_score = nn_clf_pca_10.score(X_test_pca_10, y_test)
pca_10_test_total_time = time.time() - pca_10_test_start_time

print(f"Training finished in {pca_10_train_total_time} seconds")
print(f"Score: {pca_10_score}. Scoring took {pca_10_test_total_time} seconds")


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
y_pred = nn_clf_pca_10.predict(X_test_pca_10) 
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm.diagonal()


from time import time
start_time = time() 
for pt in X_test_pca_10: 
    nn_clf_pca_10.predict([pt])
total_time = time() - start_time 

time_for_inference = total_time / len(X_test) 
print(time_for_inference)


nn_clf = MLPClassifier(
    hidden_layer_sizes = (10, 8, 5), 
    activation = 'relu',
    solver = 'adam',
    alpha = 0.001,
    learning_rate_init = 0.001,
    tol = 0.0001,
    n_iter_no_change=10,
    max_iter = 10000
)

orig_train_start_time = time.time() 
nn_clf.fit(X_train, y_train)
orig_train_total_time = time.time() - orig_train_start_time

orig_test_start_time = time.time()
orig_score = nn_clf.score(X_test, y_test)
orig_test_total_time = time.time() - orig_test_start_time

print(f"Training finished in {orig_train_total_time} seconds")
print(f"Score: {orig_score}. Scoring took {orig_test_total_time} seconds")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
y_pred = nn_clf.predict(X_test) 
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm.diagonal()


def plot_comparison(x, y, x_label, y_label, title, colors):
    plt.clf() 

    plt.bar(x, y, color = colors)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

x = ['PCA_10', 'Original']

plot_comparison(x, [pca_10_score, orig_score], "Type Of Dataset", "Accuracy Score", "Dataset Type vs Accuracy", ["blue", "green"])
plot_comparison(x, [pca_10_test_total_time, orig_test_total_time], "Type Of Dataset", "Time for Inference", "Dataset Type vs Testing Time", ["blue", "green"])
plot_comparison(x, [pca_10_train_total_time, orig_train_total_time], "Type Of Dataset", "Time to Train", "Dataset Type vs Training Time", ["blue", "green"])
