import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt


# Function to detect and crop faces using MTCNN
def detect_and_crop_faces(image_path, detector):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            return []
        # Convert the image from BGR (OpenCV format) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = detector.detect_faces(image_rgb)

        cropped_faces = []
        for face in faces:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)
            cropped_face = image_rgb[y:y + height, x:x + width]
            cropped_faces.append(cropped_face)

        return cropped_faces
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


# Initialize MTCNN face detector
detector = MTCNN()

# Path to a single image
image_path = 'Photo/Celeba_Test/005496.jpg'

cropped_faces = detect_and_crop_faces(image_path, detector)

if cropped_faces:
    # Display one of the cropped faces
    plt.imshow(cropped_faces[0])
    plt.axis('off')
    plt.show()
else:
    print("No faces detected.")
