import os
import torch
import torch.nn as nn
from PIL import Image
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

# Path to the folder containing images
folder_path = "D:/Segmentation And Classification/Output"
image_files = os.listdir(folder_path)

# Displaying an image of Class : Bottle Beer 320 ML
image_bot_beer_320_ml = Image.open(r"D:/Segmentation And Classification/Output/Bot Beer 320/Bot_Beer_320ML_0b247e03-b75c-ed11-9562-000d3a85602e_3_masked_0.jpg")
image_bot_beer_320_ml.show()

# Displaying an image of Class : Bottle Beer 320 ML
image_bot_beer_620_ml = Image.open(r"D:/Segmentation And Classification/Output/Bot Beer 620/Bot_Beer_620ML_00c1afb0-ddc6-ed11-b597-000d3a82c248_2_masked_0.jpg")
image_bot_beer_620_ml.show()

# Function to process a single image
def process_image(image_path, model, preprocess, weights):
    batch = preprocess_image(image_path, preprocess)
    prediction = get_predictions(model, batch)
    class_id = prediction.argmax().item()
    category_name = weights.meta["categories"][class_id]
    return category_name


# Step 1: Initialize model with the best available weights
# model = resnet50(pretrained=True)
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
for parameter in model.parameters(): 
    parameter.requires_grad = False
num_features = model.fc.in_features
num_classes = 2
model.fc = nn.Linear(num_features , num_classes)
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

with open("classification Results Beer Bootle 320 ML.txt", "w") as file:
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



