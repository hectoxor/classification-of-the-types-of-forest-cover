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
