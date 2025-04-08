# Skin-Cancer-Detection-System
"A deep learning-based system for the automated classification of skin lesions as benign or malignant using a Streamlit web application."

# Skin Cancer Detection System

This project develops an automated system for the early detection of skin cancer using deep learning. It utilizes a Convolutional Neural Network (CNN) to classify skin lesions as either benign or malignant, with a focus on creating an accessible tool for healthcare professionals and individuals, especially in underserved areas. [cite: 112, 513, 514, 515, 516, 36, 37, 38, 39, 40]

## Key Features

* **Binary Classification:** CNN classifies skin lesions as benign or malignant. [cite: 57, 58, 59, 60, 40, 572, 573, 574, 575]
   
* **Image Analysis:** Processes digital images of skin lesions to identify potential indicators of malignancy. [cite: 11, 12, 13, 14, 15]
   
* **Streamlit Web Application:** Provides a user-friendly interface for image upload and prediction. [cite: 24, 25, 26, 27, 106, 107, 108, 109, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567]
   
* **Data Augmentation:** Employs data augmentation to enhance model robustness. [cite: 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 138, 534, 535, 536, 537, 538, 539, 540, 579, 580]

## Files Description

* `main.py`: This Python script contains the code for the Streamlit web application. It provides the user interface for uploading skin lesion images and displaying the CNN's predictions. [cite: 106, 107, 108, 109, 166, 167, 168, 24, 25, 26, 27, 613, 614, 619, 620, 621, 622, 623]
* `.ipynb file`: This Jupyter Notebook file contains the code for building, training, and evaluating the Convolutional Neural Network (CNN) model. It includes data preprocessing, model architecture definition, training procedures, and evaluation metrics. [cite: 606, 607, 608, 609, 610, 611, 612, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 65, 66, 67, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 257, 258, 259, 260, 261, 262, 263, 74, 75, 76, 77, 78, 79, 80, 81, 82]

## Technologies Used

* Python [cite: 218, 219, 220, 221, 610]
* TensorFlow and Keras [cite: 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 611, 612]
* Streamlit [cite: 24, 25, 26, 27, 613, 614]
* NumPy [cite: 253, 254, 255, 256, 615, 616]
* Matplotlib [cite: 257, 258, 259, 260, 261, 262, 263, 617, 618]

## Setup and Installation

* Create an environment variablein your IDE
  How to Create Environment Variables in PyCharm:
    *Go to "Run" > "Edit Configurations...".
    *Select your run/debug configuration from the list on the left.
    *In the configuration settings on the right, find the "Environment variables" section.
    *Click the "..." button next to the "Environment variables" field.
  In the "Environment Variables" dialog, you can:
    *Add new variables by clicking the "+" button and entering the "Name" and "Value".
    *Edit existing variables by selecting them and changing their values.
    *Remove variables by selecting them and clicking the "-" button.
    *Click "OK" to save the environment variables for that specific run/debug configuration.
  
* Install streamlit and tensorflow using pip install streamlit tensorflow 
* To run the Streamlit application (`streamlit run main.py`)

## Model Architecture (from .ipynb)

The CNN architecture is a sequential model consisting of: [cite: 83, 84, 85, 86, 87, 182, 183, 184, 185, 186, 187, 188, 189, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 65, 66, 67]

* Three convolutional blocks, each with:
   * Conv2D layer with ReLU activation (number of filters increasing from 32 to 64 to 128) [cite: 182, 183, 184, 185, 235, 607]
   * MaxPooling2D layer [cite: 186, 187, 188, 189, 236, 237, 607]
   * Flatten layer [cite: 190, 191, 238, 239]
   * Dense layer with 512 neurons and ReLU activation [cite: 240, 241, 607]
   * Dropout layer (50%) [cite: 242, 243, 244, 607]
* Final Dense layer with sigmoid activation for binary classification [cite: 240, 241, 607]

## Data Preprocessing (in .ipynb and main.py)

* Resizing to 224x224 [cite: 135, 136, 169, 170, 171, 172, 528, 529, 530, 607, 135, 136, 169, 170, 171, 172]
* Normalization (scaling pixel values to 0-1) [cite: 101, 102, 103, 104, 105, 137, 173, 174, 175, 176, 177, 178, 531, 532, 533, 607]
* Data Augmentation (rotation, shifting, zooming, shearing, flipping) [cite: 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 138, 534, 535, 536, 537, 538, 539, 540, 579, 580]

## Web Application Usage (main.py)

1.  Upload a skin lesion image through the Streamlit interface. [cite: 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 106, 107, 108, 109, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 619, 620, 621, 622, 623]
2.  The image is preprocessed and fed to the CNN model. [cite: 28, 29, 30, 31, 561, 562, 563]
3.  The model predicts whether the lesion is benign or malignant and provides a probability score. [cite: 32, 33, 34, 563, 564, 565, 621]
4.  The results (image, prediction, probability) are displayed in the interface. [cite: 34, 564, 565, 566, 621]

## Limitations

* This system is for preliminary assessment only and is not a substitute for professional medical diagnosis. [cite: 329, 330, 331, 332, 333, 334, 335, 394, 395, 53, 594, 632, 633, 634, 635, 636, 637]
* The model's performance depends on the quality and diversity of the training data. [cite: 630, 631]

