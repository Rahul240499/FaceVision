# Facial Recognition System from Scratch with ResNet-50 and CASIA-WebFace

🚀 **Welcome to the Facial Recognition System Project!**

This repository showcases a facial recognition system built entirely from scratch using the ResNet-50 architecture and the CASIA-WebFace dataset. Dive deep into the world of deep learning by exploring how to develop, train, and optimize a model without relying on pre-trained weights.

## 🔍 Features

- **Data Transformation**: Converted the CASIA-WebFace dataset from MXNet format to JPEG, organizing images into class-specific folders for seamless training and testing.
- **Training from Scratch**: Implemented the ResNet-50 model without using any pre-trained weights, providing a pure approach to model training.
- **Performance Optimization**:
  - **Learning Rate Scheduler**: Dynamically adjusted the learning rate during training to enhance convergence and model performance.
  - **Early Stopping**: Prevented overfitting by halting training when performance on a validation set stopped improving.
- **Libraries Used**:
  - **PyTorch**: For building and training the deep learning model.
  - **NumPy**: For numerical computations and data manipulation.
  - **Scikit-learn**: For data splitting, evaluation metrics, and additional utilities.
  - **tqdm**: For displaying progress bars during training loops.

## 📂 Repository Contents

- **Data Preprocessing Scripts**: Tools for converting and organizing the dataset.
- **Model Architecture**: Custom implementation of ResNet-50.
- **Training Pipeline**: Scripts to train the model with configurable parameters.
- **Evaluation Metrics**: Methods to assess model accuracy and performance.
- **Utilities**: Helper functions and modules to streamline the workflow.

## 🎯 Goals

- To understand the end-to-end process of building a facial recognition system.
- To learn how to train a deep learning model from scratch without pre-trained weights.
- To optimize model performance using advanced techniques like learning rate scheduling and early stopping.

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   https://github.com/Rahul240499/FaceVision.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy scikit-learn tqdm
   ```\
3. **Download the Original Dataset**:
   - Download the dataset from kaggle in the given MXNet format.
   - [Dataset](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface).
4. **Prepare the Dataset**:
   - Use the provided scripts to convert and organize the CASIA-WebFace dataset.
5. **Train the Model**:
   - Run the training script to start training the ResNet-50 model from scratch.
6. **Evaluate**:
   - Assess the model's performance using the evaluation scripts.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Embark on a journey to build a facial recognition system from the ground up and gain hands-on experience with deep learning and model optimization techniques!
