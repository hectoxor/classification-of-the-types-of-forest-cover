from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
y_pred = nn_clf_pca_10.predict(X_test_pca_10) 
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm.diagonal()
