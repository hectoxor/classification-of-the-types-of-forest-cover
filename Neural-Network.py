def nn_objective(trial):

    

    #hidden_layer_spec = (15,15,15)

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

    
    y_pred = model.predict(X_test_pca_10)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(nn_objective, n_trials=50)


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

