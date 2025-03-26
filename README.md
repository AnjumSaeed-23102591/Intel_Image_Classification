# Intel Image Classification with Transfer Learning (ResNet50)
🌍 Project Overview
This project is a tutorial-based exploration of image classification using ResNet50 and transfer learning. It focuses on training, fine-tuning, and evaluating a deep learning model on the Intel Image Classification dataset, which consists of natural scenes like buildings, forests, mountains, glaciers, streets, and seas.

The tutorial demonstrates:

Data exploration & visualization

Augmentation techniques (Rotation, CLAHE, Flipping, Zoom, Blur)

Transfer learning using ResNet50

Fine-tuning pre-trained layers

Model evaluation (accuracy, precision, recall, F1-score)

Grad-CAM for model interpretability

🎓 Educational Goal: This notebook is part of a machine learning coursework assignment to help others understand the practical implementation of transfer learning, model diagnostics, and accessible deep learning workflows.

📁 Dataset
Dataset: Intel Image Classification

Source: Kaggle Dataset Link

Classes: buildings, forest, glacier, mountain, sea, street

The dataset is organized in:


intel_image_classification/
├── seg_train/
├── seg_test/
├── seg_pred/

⚙️ How to Run
Open the notebook in Google Colab
Mount Google Drive and upload the dataset to your Drive

Follow the code sections:

Data Loading & Exploration

Augmentation Preview

Model Building (ResNet50)

Training & Fine-Tuning

Evaluation Metrics

Grad-CAM Visualization

🔍 Techniques & Tools Used
TensorFlow / Keras

ResNet50 for transfer learning

ImageDataGenerator for real-time augmentation

Custom augmentation visual previews

EarlyStopping, ModelCheckpoint

Grad-CAM interpretability

📊 Evaluation
The model is evaluated using:

Accuracy

Precision / Recall / F1-Score

Confusion Matrix

Grad-CAM Heatmaps for insight into model reasoning

📚 License
This project is shared under the MIT License.
Feel free to use, modify, and share with attribution.

🙌 Acknowledgements
Kaggle Intel Dataset

TensorFlow/Keras documentation
