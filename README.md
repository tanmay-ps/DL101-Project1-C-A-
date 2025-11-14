# DL101-Project1-C-A
# AI vs. Real Image Classifier

This project is a Deep Learning model built to classify images as either "Real" (photographs) or "AI-Generated" (created by models like DALL-E, Midjourney, or Stable Diffusion).

The model is a Convolutional Neural Network (CNN) built using Keras and TensorFlow.

## 🚀 Features

* **Model:** A sequential CNN model with 4 convolutional layers and 3 dense layers.
* **Performance:** The baseline model achieves **~94% accuracy** on a test set of 20,000 images.
* **Techniques:**
    * **Dropout** layers are used to prevent overfitting.
    * An updated model architecture (`AIGeneratedModel_Updated.h5`) is also included, which uses **Batch Normalization** for faster and more stable training.
    * The final notebook also includes **Data Augmentation** (rotation, shifting, zooming, flipping) to further improve model generalization.

## 💾 Dataset

The model was trained on a combined dataset of approximately **100,000 images** (50k real, 50k AI-generated).

The data was sourced from:
1.  **pygoogle_image:** A script was used to download initial seed images from Google Images.
2.  **CIFAKE:** A large-scale dataset (CIFAR-10 vs. AI-generated images).
3.  **Kaggle:** The "Ai Generated Images | Images Created using Ai" dataset.

All images were preprocessed by resizing them to 48x48 pixels and normalizing their pixel values.

## 🛠️ How to Use

### 1. Prerequisites
You will need the following libraries installed:
* TensorFlow / Keras
* OpenCV (`opencv-python`)
* NumPy
* Matplotlib
* scikit-learn
* (Optional) `pygoogle_image` (if you want to download new images)

You can install them via pip:
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn pygoogle_image
