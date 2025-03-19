## **Specific Test VI - Foundation Model**  

This folder contains my solution for **Specific Test VI: Foundation Model** of the DeepLense GSoC 2025 project. The task involves pretraining a **Masked Autoencoder (MAE)** on strong lensing images and fine-tuning it for **multi-class classification** and **super-resolution** using **PyTorch**.  

### 📌 **Task Overview**  
The test consists of two main parts:  
1. **Pretraining a Masked Autoencoder (MAE)** on **no_sub** samples to learn meaningful feature representations.  
2. **Fine-tuning the MAE**:  
   - For **multi-class classification** (distinguishing between no_sub, cdm, and axion).  
   - For **super-resolution** (upscaling low-resolution images using high-resolution ground truths).  

#### 📷 Sample Images for Each Task
- **Samples for multi-class classification**
   ![Sample Images](assets/classification/classSample.png)
 
- **Samples for super-resolution**
   ![Sample Images](assets/superresolution/superRsample.png)

### 📂 **Folder Structure**  
```
specific_test_06/
│── models/                        # 📂 Model definitions & weights
│   ├── mae.py                      # MAE model
│   ├── classifier.py                # Classification model
│   ├── super_resolution.py         # Super-Resolution model
│   ├── checkpoints/                 # Trained weights
│       ├── mae.pth
│       ├── classifier.pth
│       ├── super_resolution.pth
│
│── scripts/                        # 📂 Training & evaluation scripts NOTE the parameters here are hardcoded
│   ├── train_mae.py                 # Train MAE
│   ├── train_classifier.py          # Train classification model
│   ├── train_superresolution.py           # Train super-resolution model
│   ├── evaluate.py                  # Compute MSE, SSIM, PSNR, LPIPS # not created yet
│   ├── infer.py                     # Run inference on new images # not ready
│   ├── infer_01.py                  # Run inference on new images Classification # not ready
│
│── utils/                          # 📂 Helper functions
│   ├── Dataset.py                    # Data loading & augmentation
│   ├── metrics.py                    # SSIM, PSNR, LPIPS calculations
│   ├── helpful.py                    # helpful functions that's used alot
│   ├── vis.py                        # save plots like pca and tsne
│   ├── extract_encoderPart.py        # take parts from the trained mae model to be used for fine-tuning models
│
│── assets/                        # 📂 Store evaluation results
│   ├── mae/                                   # Images
│   ├── classification/                        # Images
│   ├── superresolution/                       # Images
│
│── notebooks/                      # 📂 Jupyter notebooks
│   ├── mae_training.ipynb                     # Training MAE step-by-step
│   ├── classification_training.ipynb          # Fine-tuning classifier
│   ├── super_resolution_training.ipynb        # Fine-tuning super-resolution
│
│── requirements.txt                 # 📜 Dependencies
│── README.md                         # 📜 Project overview
│── .gitignore                        # 🚫 Ignore large files (checkpoints, datasets)
```

### **Prepate Data for Masked Autoencoder (MAE) Pretraining**  

#### **Input for Encoder**
- **Sample for splitted-image**
   ![Sample Images](assets/mae/splitted_image.png)

- **Sample for masked-image**
   ![Sample Images](assets/mae/masked_image.png)

- **Masked pathces and Visible patches**

   | ![Masked Image](assets/mae/masked_patches.png) | ![EncoderInput](assets/mae/visible_patches.png) |
   |------------|-------------|


### 🛠 **Model and Approach**  
#### **1️⃣ Masked Autoencoder (MAE) Pretraining**
- **Goal:** Learn a feature representation of strong lensing images.  
- **Architecture:** Vision Transformer (ViT) backbone with a reconstruction head.  
- **Pretraining Loss:** Mean Squared Error (MSE)
- **Optimizer:** AdamW 
- **Batch Size:** *256*
- **Epochs:** *250*

#### **2️⃣ Fine-Tuning for Multi-Class Classification**
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** AdamW 
- **Batch Size:** *256*
- **Evaluation Metrics:** AUC Score, Accuracy  
- **Epochs:** *250*

#### **3️⃣ Fine-Tuning for Super-Resolution**
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** *256*
- **Evaluation Metrics:** MSE, SSIM, PSNR  
- **Epochs:** *200*
- **NOTE** The Decoder used here is not suitable for images and especially for super-resolution tasks. So we need more work on the architecture

### 📊 **Results**  
Below are the evaluation results for each task:  

#### **1️⃣ Masked Autoencoder (MAE) Pretraining**  
- **Training Loss (MSE) over 250 epochs**  
  ![MAE Loss](assets/mae/MAE_Losses.png)  
- **PCA and TSNE on the embedding**  
  - Hidder representation:  
    | ![pca](assets/mae/pca_plot.png) | ![tsne](assets/mae/tsne_plot.png) |
    |------------|------------|

#### **2️⃣ Multi-Class Classification**  
- **Accuracy & AUC Score over epochs**  
    | ![Accuracy Metrics](assets/classification/Accuracies.png) | ![AUC Metrics](assets/classification/AUC.png) |
    |------------|------------|

- **Classification Report**
```
              precision    recall  f1-score   support

      no_sub       0.97      0.99      0.98      2945
       axion       0.98      0.97      0.97      2990
         cdm       0.97      0.95      0.96      2976

    accuracy                           0.97      8911
   macro avg       0.97      0.97      0.97      8911
weighted avg       0.97      0.97      0.97      8911
```

- **Confusion Matrix and ROC Curve**
    | ![ROC Metrics](assets/classification/ROC_curve.png) | ![Confusion Matrix](assets/classification/confusion_matrix.png) |
    |------------|------------|

- **PCA & tsne plotting**
    | ![PCA](assets/classification/pca_plot.png) | ![tsne](assets/classification/tsne_plot.png) |
    |------------|------------|

#### **3️⃣ Super-Resolution**
- **MSE as a loss, SSIM, PSNR over epochs**
      | ![SSIM](assets/superresolution/SSIM.png) | ![PSNR](assets/superresolution/PSNR.png) |
    |------------|------------|
  ![MSE](assets/superresolution/MAE_Losses.png)

- **Final Metrics** *these results from best SSIM model **superresolution_SSIM** and it's very close to **superresolution_PSNR***
     - Final Validation MSE: 0.002293
     - Final Validation PSNR: 29.62
     - Final Validation SSIM: 0.9190
##### **Interpretation**

- Lower **MSE** means better reconstruction (less error).
- If MSE = 0, the images are identical.

- Higher **PSNR** means better quality.
- **Typical values:**
  - **30-50 dB** → Good quality
  - **20-30 dB** → Moderate quality
  - **<20 dB** → Poor quality
- If PSNR → ∞, it means the images are **identical** (MSE = 0).

- **SSIM = 1** → Identical images.
- **SSIM close to 0** → No structural similarity.
- Unlike MSE and PSNR, **SSIM aligns more with human perception**.

- **Super-resolution comparison**  
  - Low-res, predicted high-res, and ground truth  
    | ![LR](assets/superresolution/lr_image.png) | ![Predicted](assets/superresolution/superResoluted.png) | ![HR](assets/superresolution/hr_image.png) |
    |------------|------------|-------------|

### 🚀 **Running the Code**  
1. Open any `*.ipynb` in Jupyter Notebook.
2. Run all cells to train the models.
3. Model checkpoints will be saved in `*.pth`.

### 📬 **Submission Details**
This task is part of my DeepLense GSoC 2025 submission.