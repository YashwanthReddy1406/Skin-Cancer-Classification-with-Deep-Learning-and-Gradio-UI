# Skin-Cancer-Classification-with-Deep-Learning-and-Gradio-UI
An AI-powered skin lesion classification system that uses deep learning to detect and classify seven types of skin cancer from dermatoscopic images. Built with EfficientNetB0 and deployed with an interactive Gradio interface.
📋 Overview
Skin cancer is one of the most common types of cancer worldwide. Early detection is crucial for successful treatment. This project leverages state-of-the-art deep learning techniques to automatically classify skin lesions into seven categories, potentially assisting healthcare professionals in early diagnosis.
🎯 Key Features

7 Lesion Type Classification: Melanoma, Melanocytic nevi, Basal cell carcinoma, Actinic keratoses, Benign keratosis, Dermatofibroma, Vascular lesions
Transfer Learning: Uses pre-trained EfficientNetB0 architecture for high accuracy
Interactive Web Interface: Real-time predictions with Gradio UI
Confidence Scores: Displays probability distribution across all classes
Data Augmentation: Robust training with advanced preprocessing techniques
Class Imbalance Handling: Weighted loss function for balanced learning
Google Colab Ready: Complete notebook for easy execution

🚀 Demo
Upload a dermatoscopic image → Get instant classification → View confidence scores and clinical information
[Live Demo Link] (Add your Gradio public URL here)
📊 Dataset
HAM10000 (Human Against Machine with 10000 training images)

10,015 dermatoscopic images
7 different diagnostic categories
Collected from different populations
Acquired with different modalities

Class Distribution:

Melanocytic nevi (nv): 6,705 images
Melanoma (mel): 1,113 images
Benign keratosis (bkl): 1,099 images
Basal cell carcinoma (bcc): 514 images
Actinic keratoses (akiec): 327 images
Vascular lesions (vasc): 142 images
Dermatofibroma (df): 115 images

🛠️ Technologies Used
Backend

Python 3.8+
TensorFlow 2.x / Keras - Deep learning framework
EfficientNetB0 - Pre-trained CNN model
NumPy & Pandas - Data manipulation
Scikit-learn - Model evaluation
Pillow (PIL) - Image processing

Frontend

Gradio - Interactive web interface
Matplotlib & Seaborn - Data visualization

Development Environment

🔧 Installation & Setup
Option 1: Google Colab (Recommended)

Open the notebook in Google Colab
Run all cells sequentially
The notebook will automatically:

Install dependencies
Download the dataset
Train the model
Launch Gradio interface

🎓 Model Architecture
Base Model: EfficientNetB0

Pre-trained on ImageNet
5.3M parameters
Input size: 224×224×3

Training Configuration:

Optimizer: Adam (lr=0.001)
Loss: Categorical Cross-Entropy with class weights
Metrics: Accuracy, AUC
Data Augmentation: Rotation, flips, zoom, shifts
Callbacks: EarlyStopping, ReduceLROnPlateau

📈 Model Performance

Test Accuracy: ~85%* 
Test AUC: ~0.90* 
Training Time: ~20-25 minutes on Colab GPU

Performance metrics may vary based on training run

🔬 Methodology

Data Preprocessing

Image resizing to 224×224 pixels
Pixel normalization (0-1 range)
Stratified train-validation-test split


Data Augmentation

Random rotation (±20°)
Horizontal/vertical flips
Width/height shifts (±20%)
Zoom (±20%)


Transfer Learning

Load EfficientNetB0 with ImageNet weights
Freeze base layers
Add custom classification layers


Class Imbalance Handling

Compute class weights
Apply weighted loss function


Model Training

Batch size: 32
Epochs: 20 (with early stopping)
Learning rate reduction on plateau


Evaluation

Test on held-out dataset
Confusion matrix analysis
Per-class accuracy metrics



📊 Visualizations
The project includes comprehensive visualizations:

Class distribution analysis
Sample images from each category
Training/validation accuracy curves
Training/validation loss curves
AUC progression
Confusion matrix

⚠️ Important Disclaimer
This system is intended for educational and research purposes only.

Not a substitute for professional medical diagnosis
Should not be used for clinical decision-making
Always consult qualified healthcare professionals
Requires validation in clinical settings before real-world use

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

🔮 Future Enhancements

 Expand dataset with additional images
 Implement ensemble learning (multiple models)
 Add explainable AI (Grad-CAM visualizations)
 Deploy to Hugging Face Spaces for permanent hosting
 Mobile application development
 Integration with DICOM medical imaging standards
 Multi-language support in UI
 Real-time webcam capture feature

📚 References

HAM10000 Dataset
EfficientNet Paper
Gradio Documentation
TensorFlow Documentation

Google Colab - Cloud-based GPU training
Kaggle Hub - Dataset access
