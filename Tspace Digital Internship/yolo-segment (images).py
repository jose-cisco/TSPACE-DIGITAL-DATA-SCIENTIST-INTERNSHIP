from ultralytics import YOLO
import numpy as np
import cv2

# Load The YOLOv8 Model
model = YOLO('yolov8l-seg.pt')

# Define path to the image file
img = 'D:/Segmentation And Classification/Datasets/0_Bot_Beer_320ML/Bot_Beer_320ML_0d100ea4-d764-ee11-8df1-002248ecf787_2.png'
results = model.predict(img)

for result in (results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='result.jpg')

# len(masks)

for i in range(len(masks)): #iterating through each mask
    if(results[0].masks is not None):
        # Convert mask to single channel image
        mask_raw = results[0].masks[i].cpu().data.numpy().transpose(1, 2, 0)
        
        # Convert single channel grayscale to 3 channel image
        mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

        # Get the size of the original image (height, width, channels)
        h2, w2, c2 = results[0].orig_img.shape
        
        # Loop through each mask
        # for mask in results[0].masks:
            # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
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

            # Save the masked image
        cv2.imwrite(f"masked_image.jpg", masked)

    
