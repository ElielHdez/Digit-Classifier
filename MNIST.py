# This script trains an optimized Random Forest Multiclass model to classify digits from a data augmented MNIST dataset. The dataset is downloaded through Scikit-Learn's fetch_mldata() function at the beginning of the script and cached for later usage
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.interpolation import shift
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_mldata
from sklearn.externals import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# The MNIST is downloaded and cached for later usage
mnist = fetch_mldata('MNIST original')
attributes = mnist.data
labels = mnist.target

# The dataset is split and shuffled
X_train, X_test, y_train, y_test  = attributes[0:60000], attributes[60000:], labels[0:60000], labels[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index],y_train[shuffle_index]

# Custom Data Augmentation Transformer. Adds shifted images to the training set
class DataAugmentation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_augmented = [image for image in X]
        #for shift_vector in [[0,-1],[0,1,,[-1,0],[1,0]]]:
        for shift_vector in [[0,-1],[0,1]]:
            for image in X:
                image_copy = image.copy().reshape(28,28)
                shift(image_copy, shift_vector, cval=0)
                X_augmented.append(image_copy.reshape([-1]))
        return np.array(X_augmented)

# Data prep pipeline. Standardization included
pipeline = Pipeline([
    ("augmenter", DataAugmentation()),
    ("scaler", StandardScaler()),
])

# The attributes train set is run through the pipeline
X_train_prepared = pipeline.fit_transform(X_train.astype(np.float64))
# y_train_prepared accounts for the shifted images in X_train_prepared
y_train_prepared = np.array([label for label in y_train]*3)
# Cross-validation is used to train and test various Random Forests using Grid Search
param_grid = [
    {"n_estimators": [3,10,30], "max_features":[4,6,8,12],},
    ]
rf_clf = RandomForestClassifier(random_state=42)
search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=4)
search.fit(X_train_prepared, y_train_prepared)
# The most performant model is saved into the estimator variable
estimator = search.best_estimator_
# Uncomment to save model
joblib.dump(estimator, "forest_model.pkl")
#estimator = joblib.load("forest_model.pkl")

# A predictions vector is computed against the training_set. 100% accuracy
# predictions = estimator.predict(X_train_prepared)

# A predictions vector is computed against the test_set. 92% accuracy
predictions = estimator.predict(StandardScaler().fit_transform(X_test))

# Prints out various performance scores
print(accuracy_score(y_test, predictions))
print(f1_score(y_test, predictions, average="macro"))
print(precision_score(y_test, predictions, average="macro"))
print(recall_score(y_test, predictions, average="macro"))

# Uncomment to perform Confusion Matrix Graphical Error Analysis
# The plot shows gray/white areas, representing misclassified instances
conf_mat = confusion_matrix(y_test, predictions)
row_sums = conf_mat.sum(axis=1,keepdims=True)
print(conf_mat)
norm_conf_mat = conf_mat/row_sums
np.fill_diagonal(norm_conf_mat, 0)
plt.matshow(norm_conf_mat, cmap=plt.cm.gray)
plt.show()