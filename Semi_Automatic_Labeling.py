from deepface import DeepFace
import os
import pandas as pd

# Directory containing the images
image_dir = 'Photo/Celeba_Test_Detected'
# Path to the CSV file to save metadata
metadata_path = 'Photo/Celeba_Test_Detected/metadata.csv'

# Initialize the metadata DataFrame
if os.path.exists(metadata_path):
    metadata = pd.read_csv(metadata_path)
else:
    metadata = pd.DataFrame(columns=['image_path', 'predicted_nationality', 'age', 'dominant_gender', 'emotion'])


# Function to process images and update metadata
def process_images(image_dir, metadata, metadata_path):
    batch_size = 1000
    start_index = len(metadata)

    for index, image_name in enumerate(os.listdir(image_dir), start=start_index):
        if image_name.endswith(('.jpg', '.png')) and image_name not in metadata['image_path'].tolist():
            image_path = os.path.join(image_dir, image_name)
            try:
                print(f"Processing {image_name}")
                results = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race', 'emotion'])

                # Check if the result is a list (multiple faces detected)
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        nationality = result['dominant_race']
                        age = result['age']
                        gender_confidences = result['gender']
                        dominant_gender = max(gender_confidences, key=gender_confidences.get)
                        emotion = result['dominant_emotion']
                        metadata.loc[len(metadata)] = [f"{image_name}_face_{i}", nationality, age, dominant_gender,
                                                       emotion]
                else:
                    # Single face detected
                    nationality = results['dominant_race']
                    age = results['age']
                    gender_confidences = results['gender']
                    dominant_gender = max(gender_confidences, key=gender_confidences.get)
                    emotion = results['dominant_emotion']
                    metadata.loc[len(metadata)] = [image_name, nationality, age, dominant_gender, emotion]

                # Save every batch_size images
                if index % batch_size == 0:
                    metadata.to_csv(metadata_path, index=False, mode='a', header=not os.path.exists(metadata_path))

            except Exception as e:
                print(f"Error processing {image_name}: {e}")

    # Save any remaining data
    metadata.to_csv(metadata_path, index=False, mode='a', header=not os.path.exists(metadata_path))


# Process the images
process_images(image_dir, metadata, metadata_path)