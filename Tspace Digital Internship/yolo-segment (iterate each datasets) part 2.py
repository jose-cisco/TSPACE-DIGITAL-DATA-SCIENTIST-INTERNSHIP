from ultralytics import YOLO
import numpy as np
import cv2
import os

# Load The YOLOv8 Model
model = YOLO('yolov8l-seg.pt')

# Define paths to the input and output folders
input_folder = 'D:/Segmentation And Classification/Datasets/1_Bot_Beer_620ML'
output_folder = 'D:/Segmentation And Classification/Bot Beer 620'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Process each image in the input folder
for file in files:
    if file.endswith('.png') or file.endswith('.jpg'):
        # Define the path to the image file
        img_path = os.path.join(input_folder, file)
        
        # Predict using YOLOv8
        results = model.predict(img_path)

        # Process the results
        if results[0].masks is not None:
            # Convert mask to single channel image
            mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
            
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))

            # Get the size of the original image (height, width, channels)
            h2, w2, c2 = results[0].orig_img.shape
            
            # Resize the mask to the same size as the image
            mask = cv2.resize(mask_3channel, (w2, h2))

            # Convert BGR to HSV
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

            # Define range of brightness in HSV
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([0, 0, 1])

            # Create a mask. Threshold the HSV image to get everything black
            mask = cv2.inRange(mask, lower_black, upper_black)

            # Invert the mask to get everything but black
            mask = cv2.bitwise_not(mask)

            # Apply the mask to the original image
            masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)

            # Save the masked image
            output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_masked.jpg")
            cv2.imwrite(output_path, masked)

            print(f"Processed image: {file}")

print("Processing complete.")
