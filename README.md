# Classification of Tomato Plant Disease


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ShubhamPawar-3333/Classification_of_tomato_plant_disease.git">
  </a>

<h3 align="center">Classification of Tomato Plant diseases</h3>

  <p align="center">
    This project deals with the development of machine learning model based on Convolution Neural Network (CNN) for classifying the disease on tomato plants.
  </p>
  <p>
    # Tomato Disease Classification

This project focuses on building a deep learning model for classifying various diseases in tomato plants. The project utilizes TensorFlow and Keras for model development and image data augmentation. Below is a detailed guide on the project structure, data preparation, model building, training, evaluation, and prediction.

## Project Structure

- **Importing necessary libraries:** 
    - TensorFlow
    - NumPy
    - Matplotlib
    - IPython.display
    - ImageDataGenerator from Keras

- **Global Initialization Of some important variables:**
    - Image_Size: 256
    - Batch_Size: 32
    - Channels: 3
    - Epochs: 50

- **Splitting the dataset:**
    - The dataset is divided into three sets:
        - Training dataset
        - Validation dataset
        - Test Dataset

- **Data Augmentation on the fly using Keras ImageDataGenerator:**
    - Techniques include rescaling, rotation, horizontal and vertical flipping.

- **Model Building:**
    - Convolutional Neural Network (CNN) architecture using Sequential API.
    - Input shape: (256, 256, 3)
    - Output classes: 10

- **Compiling the Model:**
    - Optimizer: Adam
    - Loss function: Sparse Categorical Crossentropy
    - Metric: Accuracy

- **Training the network:**
    - Training on the augmented data for 50 epochs.

- **Plotting the Accuracy and Loss Curves:**
    - Visualizing the training and validation accuracy and loss over epochs.

- **Running prediction on a sample image:**
    - Demonstrating the model's prediction on a sample image from the test dataset.

- **Writing a function for inference:**
    - A function to perform inference on a given image using the trained model.

- **Saving the Model:**
    - The trained model is saved in the h5 format for convenient deployment and future use.

## Usage

1. **Installation of split-folders:**
    ```bash
    $ pip install split-folders
    ```

2. **Splitting the dataset:**
    ```bash
    $ splitfolders --ratio 0.8 0.1 0.1 -- ./training/PlantVillage/Tomato_disease_categories
    ```

3. **Training the Model:**
    - Execute the provided Python script for model training.

4. **Running Inference:**
    - Utilize the `predict` function to make predictions on new images.

5. **Saving the Model:**
    - The trained model is saved as `tomato_classification.h5`.

Feel free to customize the parameters, architecture, and training settings based on your specific requirements.
  </p>
</div>

# Prepare Dataset

--- Datasets was collected from :

PlantVillage datasets = https://www.kaggle.com/datasets/arjuntejaswi/plant-village

#
  
