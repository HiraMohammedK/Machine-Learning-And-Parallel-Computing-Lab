import os
import gzip
import shutil
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,
confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

# Clone the dataset repository
!git clone https://github.com/iamavieira/handwritten-digits-mnist

# Define paths to the dataset
train_path = '/content/handwritten-digits-mnist/data/train'
test_path = '/content/handwritten-digits-mnist/data/test'

def list_files(path):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

print("Contents of the train folder:")
list_files(train_path)

print("\nContents of the test folder:")
list_files(test_path)

def extract_gz_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.gz'):
                file_path = os.path.join(root, file)
                with gzip.open(file_path, 'rb') as f_in:
                    with open(file_path[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

# Extract gz files
extract_gz_files('/content/handwritten-digits-mnist')

def load_images(file_path):
    with open(file_path, 'rb') as f:
        _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        _, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels
train_images = load_images('/content/handwritten-digits-
mnist/data/train/train-images-idx3-ubyte')
train_labels = load_labels('/content/handwritten-digits-
mnist/data/train/train-labels-idx1-ubyte')
test_images = load_images('/content/handwritten-digits-
mnist/data/test/test-images-idx3-ubyte')
test_labels = load_labels('/content/handwritten-digits-
mnist/data/test/test-labels-idx1-ubyte')

# Normalize the pixel values of the images
x_train = train_images.astype('float32') / 255.0
x_test = test_images.astype('float32') / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

x_train_small, _, y_train_small, _ = train_test_split(x_train, train_labels, train_size=0.1, random_state=42)
x_val, _, y_val, _ = train_test_split(x_test, test_labels, train_size=0.1, random_state=42)

# logistic regression model
log_reg = LogisticRegression(max_iter=1000, solver='saga', 
multi_class='multinomial', n_jobs=-1)
log_reg.fit(x_train_small, y_train_small)
y_val_pred = log_reg.predict(x_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average='macro')
recall = recall_score(y_val, y_val_pred, average='macro')
f1 = f1_score(y_val, y_val_pred, average='macro')
class_report = classification_report(y_val, y_val_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", class_report)

# Display the confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot(cmap='Blues')
plt.show()

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['saga']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, 
multi_class='multinomial'), param_grid, cv=3, scoring='accuracy')
grid_search.fit(x_train_small, y_train_small)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Predict on the validation set with the best estimator
best_log_reg = grid_search.best_estimator_
y_val_pred_best = best_log_reg.predict(x_val)
accuracy_best = accuracy_score(y_val, y_val_pred_best)
precision_best = precision_score(y_val, y_val_pred_best, average='macro')
recall_best = recall_score(y_val, y_val_pred_best, average='macro')
f1_best = f1_score(y_val, y_val_pred_best, average='macro')
class_report_best = classification_report(y_val, y_val_pred_best)

print(f"Best Model Accuracy: {accuracy_best:.4f}")
print(f"Best Model Precision: {precision_best:.4f}")
print(f"Best Model Recall: {recall_best:.4f}")
print(f"Best Model F1 Score: {f1_best:.4f}")
print("\nBest Model Classification Report:\n", class_report_best)

# Display the confusion matrix for the best model
conf_matrix_best = confusion_matrix(y_val, y_val_pred_best)
disp_best = ConfusionMatrixDisplay(conf_matrix_best)
disp_best.plot(cmap='Blues')
plt.show()

# Visualize the decision boundary using PCA
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train_small)
x_val_pca = pca.transform(x_val)

# No transformation needed for labels as they are just class identifiers
y_val_pca = y_val 

log_reg_pca = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial', n_jobs=-1)
log_reg_pca.fit(x_train_pca, y_train_small)

h = .02  # step size in the mesh
x_min, x_max = x_val_pca[:, 0].min() - 1, x_val_pca[:, 0].max() + 1
y_min, y_max = x_val_pca[:, 1].min() - 1, y_val_pca.max() + 1 # Use y_val_pca here
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


Z = log_reg_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8)
scatter = plt.scatter(x_val_pca[:, 0], x_val_pca[:, 1], c=y_val, edgecolor='k', s=20, cmap=plt.cm.viridis)
plt.title('Decision Boundary of Logistic Regression (PCA reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()
