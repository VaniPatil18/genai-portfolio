# 🌟 Generative AI Portfolio

## 🔍 Introduction to Generative AI

Generative AI refers to a class of artificial intelligence models that can generate new content — such as **text, images, audio, video**, or **code** — that resembles human-created data. Unlike traditional AI systems that classify or predict existing data, generative AI **learns the patterns** of the data and **creates new meaningful content**.

---

## ⚙️ How Generative AI Works

Generative AI models learn a **probability distribution** over the training data. During inference, they sample from this distribution to generate new outputs.

### Process:

1. **Training:** The model learns from large datasets (images, text, audio, etc.).
2. **Generation:** It creates new examples that mimic the training data.

---

## 🧠 Generative AI Models

<details>
<summary>🔹 <strong>Autoencoders</strong></summary>

Designed and implemented a fully connected autoencoder neural network for unsupervised feature learning, with a focus on image compression and reconstruction. The model was trained on the Fashion MNIST dataset, enabling it to learn compact, lower-dimensional representations of grayscale fashion images. These compressed representations were then used to accurately reconstruct the original inputs, demonstrating the model’s ability to retain essential visual features while reducing dimensionality.

Architecture :
1. Input Dimension: 784 (28×28 grayscale image flattened)
2. Encoder:
    Dense Layer 1: 512 neurons with ReLU activation
    Dense Layer 2: 32 neurons (bottleneck layer)
3. Latent Space:
    32-dimensional compressed representation capturing the essential features of the input image .
4. Decoder:
   Dense Layer 1: 512 neurons with ReLU activation
   Output Layer: 784 neurons with Sigmoid activation (reshaped to 28×28)
   Loss Function: Mean Squared Error (MSE)
5. Optimizer: Adam

![Autoencoder Output](autoencoder.jpeg)

Functional Flow :
1 . Input: Grayscale fashion image (e.g., handbag) is provided as input to the encoder.
2 . Encoding: The encoder compresses the input from 784 to 32 dimensions, capturing key visual features.
3 .  Decoding: The decoder reconstructs the original image from this compressed representation.
4 . Output: A reconstructed version of the original image is generated, maintaining visual similarity while discarding redundant information.

Key Benefits and Applications  :
1. Dimensionality Reduction: Compresses high-dimensional input images into a compact latent space.
2. Feature Learning: Learns meaningful, low-dimensional representations without supervision.
3. Denoising Capability: Robust against noise, enabling cleaner reconstructions.
4. Efficient Storage & Transmission: Reduced data size makes it suitable for memory-constrained systems.
5. Foundation for Generative Models: Can be extended into advanced models such as Variational Autoencoders (VAEs).


**Output Example:**  
![Autoencoder Output](autoencoder_output.png)

🔗 [Open my Google Colab notebook](https://colab.research.google.com/drive/1VctKToXz5HnEq3hgnZk5fyVwTOx2_AJ8)


</details>

---

<details>
<summary>🔹 <strong>VAE (Variational Auto-Encoders)</strong></summary>

GANs consist of two networks — a **Generator** and a **Discriminator** — that compete in a zero-sum game to improve image generation.

**Use Cases:**
- Image synthesis
- Super-resolution
- Deepfake generation

**Output Example:**  
![GAN Output](./gan/images/output.png)

🔗 [View GAN Project](./gan/README.md)

</details>

---

<details>
<summary>🔹 <strong>CycleGAN</strong></summary>

CycleGAN enables image-to-image translation **without paired data**.

**Use Cases:**
- Photo ↔ Painting
- Horse ↔ Zebra
- Summer ↔ Winter

**Output Example:**  
![CycleGAN Output](./cyclegan/images/output.png)

🔗 [View CycleGAN Project](./cyclegan/README.md)

</details>

---

<details>
<summary>🔹 <strong>Conditional GAN (cGAN)</strong></summary>

cGANs are conditioned on input variables like class labels to generate class-specific outputs.

**Use Cases:**
- Digit generation by label
- Face synthesis from attributes
- Text-to-image synthesis

**Output Example:**  
![cGAN Output](./cgan/images/output.png)

🔗 [View Conditional GAN Project](./cgan/README.md)

</details>

---

<details>
<summary>🔹 <strong>Diffusion Models</strong></summary>

A diffusion model is a generative model that learns to create data by reversing a gradual noising process.
During training, it adds noise to data over many steps (forward process) to learn the degradation pattern.
In generation, it starts from random noise and denoises it step-by-step to produce realistic samples.
Diffusion models are known for their stability, high-quality outputs, and flexibility across data types.


The forward process is a Markov process that gradually adds Gaussian noise to training data over a series of TT timesteps, transforming structured data into pure noise.

![diffusion_model](diffusion model.jpeg)



Purpose
The forward process simulates a noise corruption trajectory, which the model learns to reverse during training. It provides a structured way to model complex data distributions through gradual degradation.

Key Advantages
Stable Training

Optimized with a simple loss (e.g., noise prediction), avoiding adversarial instability.

Theoretical Soundness

Based on probabilistic principles with a tractable and interpretable likelihood.

High-Quality Samples

Enables generation of photorealistic, diverse outputs across modalities.

No Mode Collapse

Captures full data diversity, unlike GANs.

Versatility

Adaptable to tasks like inpainting, super-resolution, and conditional generation.



**Output Example:**  
![Diffusion Output](Screenshot 2025-06-02 204913.png)
![Diffusion Output](Screenshot 2025-06-02 204959.png)

🔗 [Open my Google Colab notebook](https://colab.research.google.com/drive/1An-eHvPoHJNW9nqR2YodlKB-I3Z56toD#scrollTo=YDNH6xsx0tZr)

</details>

---

<details>
<summary>🔹 <strong>ViT (Vision Transformer)</strong></summary>

ViT splits an image into patches and applies transformer encoders on them — treating image patches like tokens in NLP.

**Use Cases:**
- Image classification
- Object detection
- Medical image analysis

**Output Example:**  
![ViT Output](./vit/images/output.png)

🔗 [View ViT Project](./vit/README.md)

</details>

---

<details>
<summary>🔹 <strong>BERT (Bidirectional Encoder Representations from Transformers)</strong></summary>

BERT is a transformer model trained on masked language modeling and next sentence prediction.

**Use Cases:**
- Sentiment analysis
- Q&A systems
- Named Entity Recognition (NER)

**Output Example:**  
![BERT Output](./bert/images/output.png)

🔗 [View BERT Project](./bert/README.md)

</details>

---

<details>
<summary>🔹 <strong>LLMs (Large Language Models)</strong></summary>

LLMs like GPT, LLaMA, and Claude are trained on billions of tokens and used for generative text tasks.

**Use Cases:**
- Text generation
- Summarization
- Translation
- Code generation

**Output Example:**  
![LLM Output](./llm/images/output.png)

🔗 [View LLM Project](./llm/README.md)

</details>

---




