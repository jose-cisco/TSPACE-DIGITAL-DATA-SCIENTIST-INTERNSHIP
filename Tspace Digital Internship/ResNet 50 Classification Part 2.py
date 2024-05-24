import os
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

# Function to load and preprocess an image
def preprocess_image(image_path, preprocess):
    img = read_image(image_path)
    batch = preprocess(img).unsqueeze(0)
    return batch

# Function to get predictions from the model
def get_predictions(model, batch):
    with torch.no_grad():
        prediction = model(batch).squeeze(0).softmax(0)
    return prediction

# Function to process a single image
def process_image(image_path, model, preprocess, weights):
    batch = preprocess_image(image_path, preprocess)
    prediction = get_predictions(model, batch)
    class_id = prediction.argmax().item()
    category_name = weights.meta["categories"][class_id]
    return category_name

# Path to the folder containing images
folder_path = "D:/Segmentation And Classification/Bot Beer 620"
image_files = os.listdir(folder_path)

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=ResNet50_Weights)
model = resnet50(pretrained=True)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Process images and collect true and predicted labels
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    true_label = os.path.splitext(image_file)[0]  # Assuming file name is the true label
    true_labels.append(true_label)
    category_name = process_image(image_path, model, preprocess, weights)
    predicted_labels.append(category_name)


# Print the true class name and predicted class name
for true_label, predicted_label in zip(true_labels, predicted_labels):
    print(f"True Class: {true_label}, Predicted Class: {predicted_label}")

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')

# Print and save classification report
report = classification_report(true_labels, predicted_labels)
print(report)

with open("classification Results Beer Bootle 620 ML.txt", "w") as file:
    file.write("Classification Results:\n")
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        file.write(f"True Class: {true_label}, Predicted Class: {predicted_label}\n")
    file.write("Classification Report:\n")
    file.write(report)

# Print accuracy, confusion matrix, precision, and recall
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


