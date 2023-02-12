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