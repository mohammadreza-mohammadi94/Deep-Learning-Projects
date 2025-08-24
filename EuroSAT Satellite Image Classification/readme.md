# EuroSAT Satellite Image Classification with CNN

This project provides an end-to-end workflow for classifying EuroSAT satellite images using a Convolutional Neural Network (CNN) in TensorFlow.

---

## 🛰 Dataset

- **EuroSAT:** 27,000 labeled images covering 10 land-use classes.
- Images are 64x64 RGB.
- Classes include: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

Dataset can be downloaded [here](http://madm.dfki.de/files/sentinel/EuroSAT.zip).

---

## 🛠 Features

- TensorFlow data pipeline for efficient loading and preprocessing.
- Data augmentation to reduce overfitting.
- CNN architecture with multiple convolutional and pooling layers.
- Softmax output for multi-class classification.
- Training visualization with learning curves (accuracy & loss).
- Model evaluation using confusion matrix and classification report.

---

## 📊 Usage

1. Download and extract the EuroSAT dataset.
2. Rename the extracted folder to `EuroSAT` and place it in the working directory.
3. Adjust parameters (image size, batch size, epochs) if needed.
4. Run the script:

```bash
python euro_sat_cnn.py
````

---

## ⚙ Configuration

* Image size: `64x64`
* Batch size: `32`
* Epochs: `20`
* Optimizer: `Adam`
* Loss: `SparseCategoricalCrossentropy`

---

## 📈 Outputs

* Training & validation accuracy and loss plots.
* Confusion matrix visualization.
* Classification report for all 10 classes.

---

## 📝 Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy, Matplotlib, Seaborn, scikit-learn

If you want, I can also create a **super minimal 1-page version** suitable for GitHub repository landing page. Do you want me to do that?
```
