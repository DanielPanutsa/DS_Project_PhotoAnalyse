import pandas as pd
import ast

# Path to the CSV file
input_csv_path = 'Photo/Celeba_Test_Detected/metadata.csv'
output_csv_path = 'Photo/Celeba_Test_Detected/metadata_1.csv'

# Read the CSV file
df = pd.read_csv(input_csv_path)

# Function to extract the dominant gender
def get_dominant_gender(gender_dict):
    gender_dict = ast.literal_eval(gender_dict)  # Convert string representation of dictionary to an actual dictionary
    dominant_gender = max(gender_dict, key=gender_dict.get)  # Get the key with the highest value
    return dominant_gender

# Apply the function to the 'gender' column and create a new 'dominant_gender' column
df['dominant_gender'] = df['gender'].apply(get_dominant_gender)

# Save the updated DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)

print("Updated CSV with dominant gender column has been saved.")