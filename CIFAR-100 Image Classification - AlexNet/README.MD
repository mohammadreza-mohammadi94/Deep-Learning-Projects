### CIFAR-100 Classification using AlexNet Architecture  

This project implements the **AlexNet architecture** to classify images from the **CIFAR-100 dataset (coarse labels)**. The model is built using TensorFlow/Keras and includes data augmentation, preprocessing, and training pipelines.

---

### **Project Overview**  
- **Objective**: Classify CIFAR-100 images into 20 coarse categories using a convolutional neural network.  
- **Architecture**: AlexNet, a deep CNN designed for image classification tasks.  
- **Dataset**: CIFAR-100 (coarse label mode) with 60,000 images (32x32 pixels each).  

---

### **Steps in the Project**  

1. **Data Preparation**:  
   - Images resized to **140x140** for better feature extraction.  
   - Labels one-hot encoded for multi-class classification.  

2. **Data Augmentation**:  
   - Random horizontal flips and random rotations applied using Keras layers.  

3. **AlexNet Model**:  
   - Designed with **5 convolutional layers**, **3 dense layers**, and **BatchNormalization** for stable training.  
   - **Dropout layers** added for regularization.  

4. **Training and Evaluation**:  
   - Compiled with **Adam optimizer** and **categorical cross-entropy loss**.  
   - Trained for 25 epochs with **64-batch size**, achieving high accuracy.  

---

### **Technologies Used**  
- **TensorFlow/Keras** for model building.  
- **OpenCV** for image resizing.  
- **NumPy** for numerical operations.  

---

### **How to Use**  

1. Clone the repository and install the dependencies:  
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```  

2. Run the script to train the model:  

3. The model's performance metrics and training logs will be displayed.  

---

### **Future Improvements**  
- Experiment with different optimizers and hyperparameters.  
- Fine-tune AlexNet on this dataset for enhanced performance.  
- Extend to CIFAR-100 fine-label mode for granular classification.  

