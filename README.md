# **DeepLense GSoC 2025 Tests**  

This repository contains my submissions for the **DeepLense GSoC 2025 evaluation tests**, including:  
- **Common Test I: Multi-Class Classification**  
- **Specific Test VI: Foundation Model**  

**NOTE: Scripts are hardcoded, I will modify them to parse the parameters and pathes from the terminal**

## 📌 **Project Overview**  
DeepLense aims to leverage machine learning for gravitational lensing research. This repository contains my implementations of the required tests using **PyTorch**.  

## 📂 **Repository Structure**  
```
deeplense25/
│── common_test_01/   # Multi-class classification task
│── specific_test_06/ # Foundation Model task
│── GSoC25_DeepLense_Tests.pdf  # Official test document
│── README.md         # This file
```

## **📝 Test Details**  

### 🔹 **Common Test I: Multi-Class Classification**  
- **Goal:** Classify images into three categories: no substructure, subhalo substructure, and vortex substructure.  
- **Model Used:** Vision Transformer (ViT)  
- **Evaluation Metrics:** AUC score, ROC curve  
- **Location:** [`common_test_01/`](common_test_01/)  

### 🔹 **Specific Test VI: Foundation Model**  
- **Goal:** Train a **Masked Autoencoder (MAE)** on no_sub samples and fine-tune it for multi-class classification and super-resolution.  
- **Steps:**  
  1. Pre-train an MAE on no_sub samples for feature learning.  
  2. Fine-tune it for classification on the full dataset.  
  3. Further fine-tune it for a **super-resolution** task.  
- **Evaluation Metrics:** AUC score (classification), MSE/SSIM/PSNR (super-resolution).  
- **Location:** [`specific_test_06/`](specific_test_06/)  

## 🚀 **Running the Code**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/WaleedAlzamil80/deeplense25.git
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebooks inside `common_test_01/` and `specific_test_06/`.  

## 📬 **Submission Details**  
To complete my submission, I will send:  
- A link to this repository  
- My CV  
- Trained models (`.pth` files)  
- Jupyter Notebooks (`.ipynb` files)  
to **ml4-sci@cern.ch** with the subject **Evaluation Test: DeepLense**.  
