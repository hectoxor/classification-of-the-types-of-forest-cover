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
