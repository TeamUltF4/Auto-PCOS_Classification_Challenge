# PCOS Identification via Computer Vision

## Overview

In this project, we aim to develop a computer vision system to identify Polycystic Ovary Syndrome (PCOS) using sonography images. PCOS is a common hormonal disorder among women of reproductive age, and early detection is crucial for effective management and treatment.

## Objective

The primary objective of this project is to leverage computer vision techniques to analyze sonography images and classify whether ovaries exhibit signs of PCOS or not. By automating the identification process, we can assist healthcare professionals in diagnosing PCOS more efficiently.

## Methodology

1. **Data Collection**: Gather a diverse dataset of sonography images depicting ovaries with and without PCOS.

2. **Preprocessing**: Clean and preprocess the images to enhance quality and remove noise.

3. **Feature Extraction**: Extract relevant features from the images, such as texture, shape, and intensity.

4. **Model Training**: Utilize machine learning or deep learning algorithms to train a classification model on the extracted features.

5. **Model Evaluation**: Evaluate the trained model's performance using appropriate metrics such as accuracy, precision, recall, and F1 score.

6. **Deployment**: Deploy the model as a standalone application or integrate it into existing healthcare systems for real-time PCOS identification.

## Technologies Used

- Python: Primary programming language
- Keras: Deep learning library for building and training neural networks
- Pandas: Data manipulation and analysis
- shutil: High-level file operations
- scikit-learn (sklearn): Machine learning library for model training and evaluation

## Potential Challenges

- **Data Availability**: Obtaining a diverse and representative dataset of sonography images may be challenging.
- **Model Interpretability**: Ensuring the interpretability of the model's decisions is crucial for clinical acceptance.
- **Regulatory Compliance**: Adhering to regulatory standards and ensuring the model meets healthcare industry requirements.

## To run model

-**Download the model file from the repository**.
-**Change the directories as required in the below code**.

```python
import os
import zipfile
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil

# Path to the ZIP file containing images
zip_path = '/content/drive/MyDrive/images/PCOSGen-test.zip'

# Path to extract the ZIP contents
extract_path = '/content/test_dataset'

# Extracting the contents of the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Path to the pre-trained model
model_path = '/content/drive/MyDrive/healthy_model1.h5'

# Loading the pre-trained model
model = load_model(model_path)

# Output folders for healthy and unhealthy images
output_path = '/content/drive/MyDrive'
healthy_folder = os.path.join(output_path, 'healthy')
unhealthy_folder = os.path.join(output_path, 'unhealthy')

# Creating output folders if they don't exist
os.makedirs(healthy_folder, exist_ok=True)
os.makedirs(unhealthy_folder, exist_ok=True)

# DataFrame to store predictions
predictions_df = pd.DataFrame(columns=['Image', 'Prediction', 'Status'])

# Iterating through images in the 'images' subdirectory
for subdir, dirs, files in os.walk(os.path.join(extract_path, 'images')):
    for image_file in files:
        image_path = os.path.join(subdir, image_file)

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)

        # Determine status based on predictions
        status = 'Healthy' if predictions[0] < 0.5 else 'Unhealthy'

        # Save the prediction and status to the DataFrame
        predictions_df = predictions_df.append({'Image': image_file, 'Prediction': predictions[0][0], 'Status': status}, ignore_index=True)

        # Organize images based on predictions
        output_folder = healthy_folder if status == 'Healthy' else unhealthy_folder
        output_file_path = os.path.join(output_folder, image_file)
        shutil.move(image_path, output_file_path)

# Save predictions to an Excel sheet
excel_output_path = '/content/drive/MyDrive/predictions_with_status2.xlsx'
predictions_df.to_excel(excel_output_path, index=False)

print("Prediction and organization completed.")
```
## To check the model scores


## Description

```python
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score, f1_score

# Load the predictions file
predictions_df = pd.read_excel('/path/to/your/predictions/file.xlsx')

# Extract ground truth labels and predicted probabilities
y_true = predictions_df['Status'].apply(lambda x: 1 if x == 'Unhealthy' else 0).values
y_pred_probabilities = predictions_df['Prediction'].values

# Set the intervals or thresholds you want to evaluate
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Calculate mAP, accuracy, and F1 score for each threshold
for threshold in thresholds:
 # Convert predicted probabilities to binary predictions based on the threshold
 y_pred_binary = (y_pred_probabilities >= threshold).astype(int)

 # Calculate mAP
 map_score = average_precision_score(y_true, y_pred_binary)

 # Calculate accuracy
 accuracy = accuracy_score(y_true, y_pred_binary)

 # Calculate F1 score
 f1 = f1_score(y_true, y_pred_binary)

 print(f'Threshold: {threshold}, Mean Average Precision (mAP): {map_score}, Accuracy: {accuracy}, F1 Score: {f1})
```

## Future Enhancements

- Incorporate additional modalities such as patient history or blood test results for improved diagnosis accuracy.
- Develop a user-friendly interface for healthcare professionals to interact with the system.
- Extend the system to perform real-time analysis of live sonography images during medical examinations.

## Conclusion

By leveraging Python and associated libraries such as Keras, Pandas, shutil, and scikit-learn, this project aims to contribute to the early detection and diagnosis of PCOS, ultimately improving healthcare outcomes for affected individuals. If you're interested in collaborating or learning more about this project, feel free to reach out!

