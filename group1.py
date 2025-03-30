import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Download latest version
dataset_path = kagglehub.dataset_download("arshid/iris-flower-dataset")

file_path = os.path.join(dataset_path, "IRIS.csv")

# Now read the CSV file using the correct file path
df = pd.read_csv(file_path)

# Categorical classification
is_categorical = {
 "sepal_length": False,
 "sepal_width": False,
 "petal_length": False,
 "petal_width": False,
 "species": True
}

# This part here imports the libraries
# And then defines a function to create the model
# The function does one-hot encoding for categorical variables, and min-max scaling for numerical ones
# One-hot encoding transforms categorical data into a numerical format suitable for machine learning algorithms by creating binary columns
# Min-max scaling transforms values to a fixed range, and handles outliers
# I do the above to ensure highest performance in most cases

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder, QuantileTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score

# Code to make a model with one-hot and min-max scaling for categorical and numerical vars respectively
def make_model (model, x_features):

  # Separate categorical and numerical features
  categorical_features = [key for key, val in is_categorical.items() if val and key in x_features]
  numerical_features = [key for key, val in is_categorical.items() if not val and key in x_features]

  # Define transformations
  preprocessor = ColumnTransformer([
      ("numerical", MinMaxScaler(), numerical_features),
      # One-hot encoding
      ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
  ])

  # Create a full pipeline
  model = Pipeline([
      ("preprocessor", preprocessor),
      ("regressor", model)
  ])

  return model

# This is required since the data is sorted
from sklearn.utils import shuffle
shuffled_df = shuffle(df, random_state=0)

# Input x features (i.e. the input features)
x_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Predictor features (i.e. the output feature)
y_feature = "species"

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = shuffled_df[x_features]

# Parameters
k_values = range(2, 15)  # Try different k values
cv_runs = 5  # Number of cross-validations

# Storage for scores
wcss_values = {k: [] for k in k_values}  # Store WCSS per fold
silhouette_values = {k: [] for k in k_values}  # Store Silhouette Scores per fold

# Cross-validation procedure
for _ in range(cv_runs):
    X_sampled = X.sample(frac=0.8, random_state=np.random.randint(1000))  # 80% of the data randomly sampled

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X_sampled)

        wcss_values[k].append(kmeans.inertia_)
        silhouette_values[k].append(silhouette_score(X_sampled, labels))

# Compute mean and standard deviation across "folds"
wcss_mean = [np.mean(wcss_values[k]) for k in k_values]
wcss_std = [np.std(wcss_values[k]) for k in k_values]

silhouette_mean = [np.mean(silhouette_values[k]) for k in k_values]
silhouette_std = [np.std(silhouette_values[k]) for k in k_values]

# Plot WCSS with error bars
plt.figure(figsize=(8, 4))
plt.errorbar(k_values, wcss_mean, yerr=wcss_std, fmt='o-', label="WCSS ± std")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Cross-Validation for K-Means (Elbow method)")
plt.legend()
plt.show()

# Plot Silhouette Score with error bars
plt.figure(figsize=(8, 4))
plt.errorbar(k_values, silhouette_mean, yerr=silhouette_std, fmt='o-', color='red', label="Silhouette Score ± std")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Cross-Validation for K-Means (Silhouette score)")
plt.legend()
plt.show()

model = make_model(MLPClassifier(hidden_layer_sizes=[64, 64, 64], max_iter=1024), x_features) #Here, we create the model

# Train-test split - we split use 20% of our dataset for testing, and 80% for training.
# We also use a random_state of 0 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    shuffled_df[x_features], shuffled_df[y_feature], test_size=0.2, random_state=0
)

model.fit(X_train, y_train) #Here, we train the model

#In the subsequent block of code, we predict and evaluate on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural network (train-test split) - Accuracy: {accuracy:.4f}")

# Cross-Validation - we try fivefold CV
cv_runs = 5

# Linear Regression with 5-fold Cross-Validation
scores_cv = cross_validate(model, shuffled_df[x_features], shuffled_df[y_feature],
                              scoring=["accuracy"], cv=cv_runs)

accuracy_mean = scores_cv["test_accuracy"].mean()
accuracy_stddev = scores_cv["test_accuracy"].std()

print(f"Neural network (Fivefold CV) - Accuracy: {accuracy_mean : .3f} ± {accuracy_stddev : .3f}")

X = shuffled_df[x_features]

inertia_values = list()
silhouette_scores = list()

num_bootstraps = 1024

for bootstrap in range(num_bootstraps):
    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(X)

    inertia_values.append(kmeans.inertia_)  # Sum of squared distances to cluster centers
    silhouette_scores.append(silhouette_score(X, labels))  # How well-separated clusters are

inertia_std = np.std(inertia_values)
silhouette_std = np.std(silhouette_scores)

# Stddev of silhouette and inertia (WCSS) to show how different initial seeds affect outcome
print(inertia_std)
print(silhouette_std)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# define class names
class_names = np.unique(y_test)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()