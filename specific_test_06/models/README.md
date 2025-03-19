# Models

This folder contains the trained models and related scripts for classification and super-resolution tasks.

## 📂 Structure

### 🔹 `checkpoints/`
This subfolder contains the saved model weights:
- **`mae.pth`** – Trained **Masked Autoencoder (MAE)** model.
- **`encoder_embedInput.pth`** – Extracted encoder and input embedding from the trained MAE model.
- **`classifier.pth`** – Fine-tuned classifier model.
- **`superresolution_PSNR.pth`** – Fine-tuned Super-resolution model, selected best model based on **Peak Signal-to-Noise Ratio (PSNR)** metric.
- **`superresolution_SSIM.pth`** – Fine-tuned Super-resolution model, selected best model based on **Structural Similarity Index (SSIM)** metric.

### 🔹 Model Training Workflow

1. **Pretraining with MAE**
   - The **Masked Autoencoder (MAE)** model was trained and saved as `mae.pth`.

2. **Extracting Encoder Blocks & Input Embedding**
   - Using `utils/extract_encoderPart.py`, the encoder blocks and input embedding layer were extracted from `mae.pth`.
   - The extracted components are stored in `encoder_embedInput.pth` used as a base to train the classifier model and the superresolution model.

3. **Fine-tuning for Downstream Tasks**
   - **Classification:** The extracted encoder was fine-tuned to train a classifier (`classifier.pth`).
   - **Super-resolution:** Two separate models were trained using the extracted encoder also:
     - `superresolution_PSNR.pth` – Trained for best **PSNR**.
     - `superresolution_SSIM.pth` – Trained for best **SSIM**.