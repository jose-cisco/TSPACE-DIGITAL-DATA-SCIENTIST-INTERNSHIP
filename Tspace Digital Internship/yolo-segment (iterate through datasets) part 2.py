from ultralytics import YOLO
import numpy as np
import cv2
import os

# Load The YOLOv8 Model
model = YOLO('yolov8l-seg.pt')

# Define path to the folder containing the images
input_folder = 'D:/Segmentation And Classification/Datasets/1_Bot_Beer_620ML/'

# Define output folder for saving processed images
output_folder = 'D:/Segmentation And Classification/Output/Bot Beer 620'

# Iterate over the 964 images
for filename in os.listdir(input_folder):
    if filename.endswith(".png"): # assuming all images are PNG format
        # Construct the full path to the image
        img_path = os.path.join(input_folder, filename)

        # Perform prediction on the image
        results = model.predict(img_path)

        # Check if masks are present and there are at least two masks
        if results[0].masks is not None and len(results[0].masks) >= 1:
            # Process and save the results for the first two masks only
            for i in range(1): # iterating through each mask
                mask = results[0].masks[i] # Get the ith mask

                # Convert mask to single channel image
                mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
                
                # Convert single channel grayscale to 3 channel image
                mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))

                # Get the size of the original image (height, width, channels)
                h2, w2, c2 = results[0].orig_img.shape
                
                # Resize the mask to the same size as the image
                mask = cv2.resize(mask_3channel, (w2, h2))

                # Convert BGR to HSV
                hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

                # Define range of brightness in HSV
                lower_black = np.array([0,0,0])
                upper_black = np.array([0,0,1])

                # Create a mask. Threshold the HSV image to get everything black
                mask = cv2.inRange(mask, lower_black, upper_black)

                # Invert the mask to get everything but black
                mask = cv2.bitwise_not(mask)

                # Apply the mask to the original image
                masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)

                # Save the masked image with unique filename
                output_filename = os.path.splitext(filename)[0] + '_masked_' + str(i) + '.jpg'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, masked)

        print(f"Processed: {filename}")

print("All images processed and saved.")
