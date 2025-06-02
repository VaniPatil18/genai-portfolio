# genai-portfolio
A collection of Generative AI projects
# 🧠 Autoencoder for Image Compression & Reconstruction

This project demonstrates how an autoencoder learns to compress and reconstruct images. Built and tested in Google Colab.

---

## 📌 Concept

Autoencoders are neural networks trained to copy their input to their output. The architecture includes:
- **Encoder**: compresses the input into a latent representation
- **Decoder**: reconstructs the input from this compressed version

---

## 🔧 Architecture

```python
Input → Dense → ReLU → Dense (encoding) → Dense → ReLU → Dense → Output
