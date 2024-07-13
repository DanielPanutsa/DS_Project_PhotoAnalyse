import os
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt


# Function to detect and crop faces using MTCNN
def detect_and_crop_faces(image_path, detector):
    try:
        print(f"Processing {image_path}")
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read image {image_path}")
            return []
        print(f"Image shape: {image.shape}")
        # Convert the image from BGR (OpenCV format) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = detector.detect_faces(image_rgb)
        print(f"Detected {len(faces)} faces")

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

# Directory containing the images
image_dir = 'Photo/Celeba_Test'
output_dir = 'Photo/Celeba_Test_Detected'
log_file_path = 'Photo/Celeba_Test/processed_images.log'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the list of processed images
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        processed_images = set(log_file.read().splitlines())
else:
    processed_images = set()

# Open log file in append mode
with open(log_file_path, 'a') as log_file:
    # Loop over all images in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            if filename in processed_images:
                print(f"Skipping already processed image: {filename}")
                continue

            image_path = os.path.join(image_dir, filename)
            cropped_faces = detect_and_crop_faces(image_path, detector)

            for i, cropped_face in enumerate(cropped_faces):
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_face_{i}.jpg")
                cropped_face_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, cropped_face_bgr)

            print(f"Processed {filename}")
            log_file.write(f"{filename}\n")
            log_file.flush()

# Optional: Display one of the cropped faces
if cropped_faces:
    plt.imshow(cropped_faces[0])
    plt.axis('off')
    plt.show()