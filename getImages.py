import cv2
import os

# Set the path to the folder where you want to save the images
save_folder = os.path.expanduser("CameraCalibration\images")

# Create the folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    cv2.imshow('Camera', img)

    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('s'):  # Press 's' to save the image
        img_path = os.path.join(save_folder, f"image{num}.png")
        cv2.imwrite(img_path, img)
        print(f"Image saved: {img_path}")
        num += 1

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
