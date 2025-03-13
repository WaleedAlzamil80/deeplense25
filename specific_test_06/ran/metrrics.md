## **1. Mean Squared Error (MSE)**
MSE measures the average squared difference between the original image and the upscaled (or reconstructed) image. It quantifies how different two images are.

### **Mathematical Formula**
\[
MSE = \frac{1}{N} \sum_{i=1}^{N} (I_{\text{orig}}(i) - I_{\text{upscaled}}(i))^2
\]
where:
- \( N \) is the total number of pixels in the image.
- \( I_{\text{orig}}(i) \) is the pixel intensity of the original image at index \( i \).
- \( I_{\text{upscaled}}(i) \) is the pixel intensity of the upscaled image at index \( i \).

### **Interpretation**
- Lower **MSE** means better reconstruction (less error).
- If MSE = 0, the images are identical.

---

## **2. Peak Signal-to-Noise Ratio (PSNR)**
PSNR measures the ratio between the maximum possible pixel intensity and the MSE. It is expressed in **decibels (dB)**, where higher values indicate better image quality.

### **Mathematical Formula**
\[
PSNR = 10 \log_{10} \left( \frac{L^2}{MSE} \right)
\]
where:
- \( L \) is the maximum pixel intensity value (e.g., **255 for 8-bit images**, **1.0 for normalized images**).
- \( MSE \) is the mean squared error.

### **Interpretation**
- Higher **PSNR** means better quality.
- **Typical values:**
  - **30-50 dB** → Good quality
  - **20-30 dB** → Moderate quality
  - **<20 dB** → Poor quality
- If PSNR → ∞, it means the images are **identical** (MSE = 0).

---

## **3. Structural Similarity Index (SSIM)**
SSIM measures the **perceived quality** of an image by considering **luminance, contrast, and structure**, rather than pixel-wise differences.

### **Mathematical Formula**
\[
SSIM(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
\]
where:
- \( \mu_x \) and \( \mu_y \) are the **mean** pixel values of images \( x \) (original) and \( y \) (upscaled).
- \( \sigma_x^2 \) and \( \sigma_y^2 \) are the **variance** (contrast).
- \( \sigma_{xy} \) is the **covariance** (measuring structural similarity).
- \( C_1 \) and \( C_2 \) are small constants to stabilize division when denominators are close to zero.

### **Interpretation**
- **SSIM = 1** → Identical images.
- **SSIM close to 0** → No structural similarity.
- Unlike MSE and PSNR, **SSIM aligns more with human perception**.

---

## **Comparison & Use Cases**
| **Metric**  | **Measures**                 | **Range**         | **Best Value** | **Usage** |
|------------|-----------------------------|------------------|--------------|------------|
| MSE        | Pixel-wise error             | \([0, \infty)\)   | 0            | Fast but naive |
| PSNR       | Signal vs. noise ratio       | \([0, \infty)\)   | High values  | Common in image compression |
| SSIM       | Perceptual similarity        | \([-1, 1]\) or \([0, 1]\) | 1 (Best) | Best for real-world quality assessment |

---

### **Summary**
- **Use MSE** when you need a simple, pixel-wise error.
- **Use PSNR** when comparing image fidelity (used in compression).
- **Use SSIM** for human-perceptual quality assessment.