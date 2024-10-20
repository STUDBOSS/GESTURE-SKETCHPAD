import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Function to read image data and map classes to labels
def read_data(path):
    classes = {'square': 0, 'circle': 1, 'triangle': 2}
    data = []
    for cls in classes:
        folder_path = os.path.join(path, cls)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            data.append([img_path, classes[cls]])
    return pd.DataFrame(data, columns=['path', 'class'])

# Function to read an image
def read(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

# Function to display an image
def show(img, title=None):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis(False)
    plt.show()

# Function to convert the image to grayscale and apply blurring
def gray_and_blurring(img, label=None):
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    if label == 0 or label == 1:  # Apply blurring for circles and squares
        gray_img = cv.blur(gray_img, (15, 15))
    return gray_img

# Function to binarize the image
def binarize(img):
    _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)
    return img_bin

# Function to find the contour of the largest object in the binary image
def contour(img_bin):
    contours, _ = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    best_contour = max(contours, key=cv.contourArea)
    return best_contour

# Function to extract the histogram of the contour directions
def histogram(img_bin):
    hist = np.zeros((8,))
    best_contour = contour(img_bin)
    code = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }

    for i in range(len(best_contour) - 1):
        x1, y1 = best_contour[i][0]
        x2, y2 = best_contour[i+1][0]
        dx = x2 - x1
        dy = y2 - y1
        hist[code[(dx, dy)]] += 1

    return hist / hist.sum()

# Function to train and save the RandomForest model
def train_model(X_train, y_train):
    Random_Forest = RandomForestClassifier(n_estimators=700, criterion='gini', max_depth=None, n_jobs=-1)
    Random_Forest.fit(X_train, y_train)

    # Save the trained model to a file
    with open('shape_model.pkl', 'wb') as f:
        pickle.dump(Random_Forest, f)

    return Random_Forest

# Function to predict the shape of an image using the trained model
def prediction(img, model):
    mapping = {0: 'square', 1: 'circle', 2: 'triangle'}
    gray_img = gray_and_blurring(img)
    bin_img = binarize(gray_img)
    hist = histogram(bin_img)
    predicted_class = model.predict([hist])[0]
    return mapping[predicted_class]

# Reading dataset
my_data = read_data('shapes')

# Data preparation
X = []
for i in range(len(my_data['path'])):
    path = my_data['path'][i]
    img = read(path)
    gray_img = gray_and_blurring(img, my_data['class'][i])
    bin_img = binarize(gray_img)
    X.append(histogram(bin_img))

X = np.array(X)
y = my_data['class']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# Training the model
best_model = train_model(X_train, y_train)

# Model evaluation
prediction_train = best_model.predict(X_train)
prediction_test = best_model.predict(X_test)

# Displaying accuracy
print("Training accuracy:", accuracy_score(y_train, prediction_train))
print("Testing accuracy:", accuracy_score(y_test, prediction_test))

# Classification report and confusion matrix
shapes_types = ['Square', 'Circle', 'Triangle']
print("\t\t\tTesting Performance Classification Report")
print(classification_report(y_test, prediction_test, labels=y_test.unique()[::-1], target_names=shapes_types))

# Plot confusion matrix
conf_mat = pd.DataFrame(confusion_matrix(y_test, prediction_test), columns=shapes_types)
conf_mat = conf_mat.rename(dict(enumerate(shapes_types)), axis=0)
plt.figure(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, fmt='g', cmap='coolwarm')
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
plt.show()

# Example of usage: Predicting the shape of an image
image_path = 'ci.png'  # Replace with your test image path
image = read(image_path)
predicted_shape = prediction(image, best_model)

# Display the image and predicted shape
show(image, f"Predicted Shape: {predicted_shape}")
print(f"The predicted shape is: {predicted_shape}")
