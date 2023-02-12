import time 

pca_10_train_start_time = time.time() 
nn_clf_pca_10.fit(X_train_pca_10, y_train)
pca_10_train_total_time = time.time() - pca_10_train_start_time

pca_10_test_start_time = time.time()
pca_10_score = nn_clf_pca_10.score(X_test_pca_10, y_test)
pca_10_test_total_time = time.time() - pca_10_test_start_time

print(f"Training finished in {pca_10_train_total_time} seconds")
print(f"Score: {pca_10_score}. Scoring took {pca_10_test_total_time} seconds")

from time import time
start_time = time() 
for pt in X_test_pca_10: 
    nn_clf_pca_10.predict([pt])
total_time = time() - start_time 

time_for_inference = total_time / len(X_test) 
print(time_for_inference)
