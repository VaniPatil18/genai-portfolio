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
A Variational Autoencoder (VAE) is a type of neural network that not only compresses data like a regular autoencoder but also learns a probabilistic latent space, allowing it to generate new data that looks like the training data (e.g., new handwritten digits).

Architecture :
1. Encoder: Converts the input (e.g., an image) into a compressed representation, but instead of a single point, it outputs a distribution (mean and variance).
2. Latent Sampling: Samples a point from this distribution using a special trick to allow training.
3. Decoder: Reconstructs the input from this sampled point.
4. Loss Function: Encourages the reconstruction to be accurate and the latent space to be well-structured.

![vae_bd](https://github.com/user-attachments/assets/a4b2c347-1fb0-4822-a96c-6718deb5d3b6)


**Output Example:**  
![vae_op](https://github.com/user-attachments/assets/5ff71441-0162-4f35-80d0-4602ca1a56b2)

🔗 [Open my Google Colab notebook](https://colab.research.google.com/drive/1vYlGMwf08j50uokxhn2C31NA5_RWlNM1)

</details>

---

<details>
<summary>🔹 <strong>CycleGAN</strong></summary>

CycleGAN is a type of GAN (Generative Adversarial Network) designed for image-to-image translation without paired data. It can learn to translate an image from one domain (e.g., horses) to another (e.g., zebras) and vice versa, without needing exact image pairs for training.

Architecture :
1. Start (Input Image) :
The model begins with an image from Domain A (e.g., a horse).

2. Generator A→B :
Transforms the horse image into an image that looks like it belongs to Domain B (e.g., a zebra). This is the Generated Image.

3. Discriminator B :
Receives both real zebra images (from actual data) and generated zebra images.
Tries to distinguish between real and fake.
Outputs True (real) or False (fake).

4. Cycle Consistency :
The generated zebra image is fed into another generator B→A, which tries to reconstruct the original horse image.
This ensures that translations are meaningful and reversible.

5. Discriminator A :
Similarly, it distinguishes between real horse images and generated ones.

![cycleganbd](https://github.com/user-attachments/assets/792f6b11-2548-4b00-a094-fd55fd1a8d15)

Advantages of CycleGAN :
1. Unpaired Data: No need for aligned image pairs (huge benefit for real-world data).
2. Bi-directional Translation: Learns both domain transformations (e.g., horse ↔ zebra).
3. Preserves Structure: Maintains content while changing style or domain.
4. Wide Applications:

    Style transfer (e.g., photo → painting)
    Object transfiguration (e.g., horse ↔ zebra)
    Medical image translation
    Season or weather change in images

**Output Example:**  
![cycleganop](https://github.com/user-attachments/assets/a39f4e68-8ad5-4584-9903-75e765699c2e)
🔗 [Open my Google Colab notebook](https://colab.research.google.com/gist/sgshrigouri/a86b0ab07c0f0d6b2eb2718406355864/cyclegan.ipynb)

</details>

---

<details>
<summary>🔹 <strong>Conditional GAN (cGAN)</strong></summary>

A Conditional GAN (cGAN) is an extension of the standard Generative Adversarial Network (GAN), where both the generator and discriminator receive additional information (often called condition or label, denoted by y) alongside their inputs.

In a basic GAN, the generator creates images from random noise z, and the discriminator learns to distinguish real from fake images. In cGAN, both models are conditioned on labels (e.g., digit class in MNIST). This allows targeted generation—you can generate specific digits like "3", "7", etc., by conditioning the model with label y.
![cganbd](https://github.com/user-attachments/assets/3f28f289-8af5-4118-b6bc-6970c01441e6)


 Architecture of Conditional GAN :
 1. Inputs: Random noise vector (z) — serves as input for the generator.
Condition label (y) — categorical info like class (e.g., 0–9 in MNIST).

3. Generator:
Takes both z and y (concatenated or embedded).
Generates a fake image conditioned on label y.

4. Discriminator:
Takes both an image (real or fake) and label y.
Learns to distinguish whether the image-label pair is:
a real image with true label (real),
or a fake/generated image with label (fake).

5. Flow:
   1. Real data and generated data are both passed to the discriminator along with the label.
   2. Discriminator tries to output:
"Real" for real image + label,
"Fake" for generated image + label.
   3. Generator learns to fool the discriminator into thinking its output is real and matches the label.

✅ Advantages of Conditional GANs:
1. Controlled Generation:
You can generate specific classes (e.g., only "7s") rather than random outputs.

3. More Data-Efficient:
Conditioning provides strong prior knowledge, reducing the space the model needs to learn.

4. Better Image Quality:
Label information helps the generator produce more realistic, class-specific details.

5. Useful for Multi-class Scenarios:
Applications like text-to-image, image-to-image translation, attribute-based generation, etc.

6. Improved Training Stability:
Adding labels provides structure to training, which can improve convergence.

🧠 Applications of cGANs:
1. Handwritten digit generation (like MNIST).

2. Image colorization (conditioned on grayscale).

3. Super-resolution (conditioned on low-res input).

4. Fashion and face synthesis with specific features.

5.Text-to-image generation (label = text embedding).

**Output Example:**  
![conditiongan_e1](https://github.com/user-attachments/assets/500d00ea-7932-4957-b5c1-64a716a9955b)
 ![conditiongane2](https://github.com/user-attachments/assets/6b8c4439-55e1-42dc-8010-5fa17ff956e8)


🔗[Open my Google Colab notebook](https://colab.research.google.com/drive/1r7JwhvIwMTWehDEFZK7fC1y7QUSJIRyB#scrollTo=L8CXgejbh13n)

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

![diffusion_model](diffusion_model_1.png)



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

![diffusion_model](diffusionimage.png)

**Output Example:**  
![Diffusion Output](noise2.png)

🔗 [Open my Google Colab notebook](https://colab.research.google.com/drive/1An-eHvPoHJNW9nqR2YodlKB-I3Z56toD#scrollTo=YDNH6xsx0tZr)

</details>

---

<details>
<summary>🔹 <strong>ViT (Vision Transformer)</strong></summary>

A Vision Transformer (ViT) is a deep learning model that applies the transformer architecture, originally designed for NLP, to image data. Instead of using convolutional layers like CNNs, ViT treats images as sequences of patches, just like words in a sentence.

Architecture of Vision Transformer (ViT) :
1. Image Preprocessing
    Input image (e.g., 224×224×3) is split into fixed-size patches (e.g., 16×16).
    Each patch is flattened and linearly embedded into a vector (e.g., 768-d).
    A [CLS] token (like in BERT) is prepended for classification.
    Positional Encodings are added to retain spatial information.
![vit1](https://github.com/user-attachments/assets/7f7056fe-69ac-4e7e-9279-259724ee4bff)
2. Transformer Encoder Blocks :
  Each block consists of:
   Layer Normalization
   Multi-Head Self-Attention (as shown in your image)
   Feed Forward Network (FFN)
   Residual Connections

3. Classification Head :
Output corresponding to the [CLS] token is passed through a linear layer for final prediction.

![vit2](https://github.com/user-attachments/assets/bdc1a545-1425-4343-b8f3-2b13fc5c2812)

Advantages of Vision Transformers Over CNNs :
1. Global Understanding: ViTs capture long-range dependencies from the start using self-attention, unlike CNNs which focus locally in early layers.
2. Scalable Efficiency: They scale better with data and model size, often outperforming CNNs when pretrained on large datasets.
3. Interpretability: Attention maps make ViTs more transparent, clearly showing which image regions influence predictions.
4. Architectural Flexibility: ViTs adapt easily to diverse tasks without needing specialized design changes.
5. Unified Modeling: Their compatibility with NLP makes ViTs ideal for multimodal models (e.g., image + text).
**Output Example:**  
![vitop](https://github.com/user-attachments/assets/bd57440d-381e-4cf5-94b9-fba757761fb5)

🔗  [Open my Google Colab notebook](https://colab.research.google.com/drive/19KwE1jz9rPKfb_3Li0TnzCyNFg6Co_mR#scrollTo=_gUMk1N4W78C)
</details>

---

<details>
<summary>🔹 <strong>BERT (Bidirectional Encoder Representations from Transformers)</strong></summary>

BERT is a deep learning model developed by Google AI in 2018 designed for Natural Language Understanding (NLU) tasks. It is based entirely on the Transformer architecture, specifically leveraging the encoder part.

What distinguishes BERT is its bidirectional context understanding, allowing it to grasp the meaning of words based on both their left and right context, unlike traditional unidirectional models.

 Architecture of BERT
1. Base Configuration
BERT comes in two standard configurations:
BERT-Base: 12 Transformer encoder layers, 768 hidden units, 12 attention heads (110M parameters).
BERT-Large: 24 Transformer encoder layers, 1024 hidden units, 16 attention heads (340M parameters).

2. Components
Input Embeddings:
Token Embeddings: Represent individual subwords or tokens.
Segment Embeddings: Distinguish between sentence A and B in pair tasks.
Position Embeddings: Encode positional information of tokens.
Encoder Stack:
Multiple layers of multi-head self-attention and feed-forward networks.


🔗  [Open my Google Colab notebook](https://colab.research.google.com/drive/1gkmTkvHYe8e8vYcaROq2OcGmn4Z2pV-P)

</details>






