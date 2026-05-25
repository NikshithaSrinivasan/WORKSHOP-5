# WORKSHOP-5
# License-Plate-Detection-using-OpenCV-and-Haar-Cascade-Classifier

## PROGRAM:
```python
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

image_path = 'my image.jpg'  # <-- Change this to your image filename
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please check the 'image_path' variable.")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Histogram Equalization for better contrast
equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image (Blur + Equalized)")
plt.axis('off')
plt.show()

cascade_path = 'haarcascade_frontalface_default.xml'

if not os.path.exists(cascade_path):
    print("Cascade file not found. Downloading...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)
    print("Cascade file downloaded successfully!")

face_cascade = cv2.CascadeClassifier(cascade_path)

faces = face_cascade.detectMultiScale(
    equalized,          # Preprocessed grayscale image
    scaleFactor=1.1,    # Scaling factor between image pyramid layers
    minNeighbors=5,     # Higher value -> fewer false detections
    minSize=(30, 30)    # Minimum object size
)

print(f"Total Faces Detected: {len(faces)}")
output = image.copy()
save_dir = "Detected_Faces"
os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face_crop = image[y:y+h, x:x+w]
    save_path = f"{save_dir}/face_{i+1}.jpg"
    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("⚠️ No faces detected. Try adjusting parameters or using a clearer image.")

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()
```

## OUTPUT:

### Original image:

<img width="165" height="294" alt="image" src="https://github.com/user-attachments/assets/29ef699c-b556-4e96-af81-c2dfb3316ac2" />


### Grayscale image:

<img width="158" height="283" alt="image" src="https://github.com/user-attachments/assets/4a70d04a-2980-4934-8a6a-040d4d7a9da1" />

### Preprocessed image:

<img width="245" height="288" alt="image" src="https://github.com/user-attachments/assets/0d0249c8-13ec-4809-97de-4bdce7720cf3" />

### Detected faces:

<img width="157" height="287" alt="image" src="https://github.com/user-attachments/assets/127e6f5a-19d3-45f2-be24-1dc946caa7c6" />


## Result:
Thus, the program was executed successfully.
