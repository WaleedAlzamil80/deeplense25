# **Self-Supervised MAE, Classification, and Super-Resolution Training Pipeline**  
🚀 **A complete deep learning pipeline** for training a **Masked Autoencoder (MAE)**, fine-tuning it for **classification**, and extending it for **super-resolution**. This project **rescales all data to `150x150` using bicubic interpolation** and evaluates upscaling quality using **MSE, PSNR, SSIM, and LPIPS**.  

---

## **📌 Project Overview**  
This repository provides a structured framework for training:  

✅ **Masked Autoencoder (MAE)** → Self-supervised pretraining using `150x150` images.  
✅ **Classification Model** → Fine-tuned on MAE’s learned features.  
✅ **Super-Resolution Model** → Trained to enhance low-resolution images (`75x75 → 150x150`).  
✅ **Evaluation Metrics** → MSE, PSNR, SSIM, and LPIPS for image quality assessment.  

**Data is first resized to `150x150` using bicubic interpolation** for consistency across tasks.

---

## **📂 Folder Structure**  
```
repo_name/
│── models/                        # 📂 Model definitions & weights
│   ├── mae.py                      # MAE model
│   ├── classifier.py                # Classification model
│   ├── super_res.py                 # Super-Resolution model
│   ├── checkpoints/                 # Trained weights
│       ├── mae.pth
│       ├── classifier.pth
│       ├── super_res.pth
│
│── scripts/                        # 📂 Training & evaluation scripts
│   ├── train_mae.py                 # Train MAE
│   ├── train_classifier.py          # Train classification model
│   ├── train_super_res.py           # Train super-resolution model
│   ├── evaluate.py                  # Compute MSE, SSIM, PSNR, LPIPS
│   ├── infer.py                      # Run inference on new images
│
│── utils/                          # 📂 Helper functions
│   ├── dataset_loader.py            # Data loading & augmentation
│   ├── metrics.py                   # SSIM, PSNR, LPIPS calculations
│   ├── visualization.py              # Plot results, generate heatmaps
│
│── results/                        # 📂 Store evaluation results
│   ├── sample_outputs/              # Sample images before & after processing
│   ├── logs/                        # Training logs
│   ├── comparison_metrics.json       # JSON file with evaluation metrics
│
│── notebooks/                      # 📂 Jupyter notebooks for analysis
│   ├── mae_training.ipynb            # Training MAE step-by-step
│   ├── classification_analysis.ipynb # Checking classifier performance
│   ├── super_resolution.ipynb        # Fine-tuning super-resolution
│
│── requirements.txt                 # 📜 Dependencies
│── README.md                         # 📜 Project overview
│── .gitignore                        # 🚫 Ignore large files (checkpoints, datasets)
```

---

## **📥 Installation & Setup**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your_username/repo_name.git
cd repo_name
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ (Optional) Install GPU-Optimized PyTorch**
Check the official **[PyTorch installation guide](https://pytorch.org/get-started/locally/)** for the correct version.

---

## **🚀 Training Workflow**  
### **1️⃣ Train the MAE Model (Self-Supervised Learning)**
```bash
python scripts/train_mae.py --epochs 50 --batch_size 32 --lr 3e-4
```
🛠 **This step pretrains the MAE on `150x150` images using masked image modeling.**  

### **2️⃣ Train the Classification Model (Fine-Tuning on MAE Features)**
```bash
python scripts/train_classifier.py --epochs 30 --batch_size 32 --lr 1e-4
```
🛠 **This step uses the MAE encoder to classify images.**  

### **3️⃣ Train the Super-Resolution Model (`75x75 → 150x150`)**
```bash
python scripts/train_super_res.py --epochs 40 --batch_size 16 --lr 2e-4
```
🛠 **The model learns to upscale images from `75x75` to `150x150`.**  

---

## **📊 Evaluation: Comparing Upscaling Quality**
Run **evaluation metrics (MSE, SSIM, PSNR, LPIPS) on test images**:
```bash
python scripts/evaluate.py --input_folder data/test/
```

---

## **📌 Model Architectures**
### **🟢 Masked Autoencoder (MAE)**
- Vision Transformer (ViT) as the encoder.  
- Trained with **masked image modeling**.  
- **Pretrained model used for classification & super-resolution.**

### **🟢 Classification Model**
- Uses **pretrained MAE encoder**.  
- Final classification **MLP head** for `N` classes.  

### **🟢 Super-Resolution Model**
- Uses a **CNN + Transformer-based decoder**.  
- Learns to **upsample low-resolution images**.  

---

## **🖼 Sample Results**
| **Method** | **Original** | **Upscaled** | **MSE ↓** | **PSNR ↑** | **SSIM ↑** | **LPIPS ↓** |
|------------|-------------|-------------|----------|----------|----------|----------|
| **Bilinear** | ![original](results/sample_outputs/original.png) | ![bilinear](results/sample_outputs/bilinear.png) | 0.0051 | 28.5 | 0.82 | 0.15 |
| **Bicubic** | ![original](results/sample_outputs/original.png) | ![bicubic](results/sample_outputs/bicubic.png) | 0.0039 | 30.2 | 0.87 | 0.10 |
| **Super-Res Model** | ![original](results/sample_outputs/original.png) | ![sr_model](results/sample_outputs/superres.png) | 0.0025 | 33.1 | 0.92 | 0.05 |

---
## **📌 References**
- **Masked Autoencoders (MAE):** [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)  
- **ViT for Image Classification:** [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)  
- **SwinIR (Super-Resolution Transformer):** [https://arxiv.org/abs/2108.10257](https://arxiv.org/abs/2108.10257)  

---

## **🛠 Future Improvements**
🔹 **Train MAE on more datasets.**  
🔹 **Test more super-resolution models (SwinIR, ESRGAN).**  
🔹 **Optimize training speed (AMP, mixed precision).**  

---
## **🤝 Contributing**
1. Fork the repo 🍴  
2. Create a feature branch (`git checkout -b feature-branch`)  
3. Commit changes (`git commit -m "Added feature"`)  
4. Push to your branch (`git push origin feature-branch`)  
5. Open a Pull Request! 🚀  

---
## **📩 Contact**
📌 **Author:** Waleed  
📌 **Email:** your_email@example.com  
📌 **GitHub:** [github.com/WaleedAlzamil8)](https://github.com/WaleedAlzamil8)
