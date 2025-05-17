# Lung Cancer Classification

A PyTorch-based pipeline for early detection of lung cancer subtypes using transfer learning (ResNet-50), class-weighted Cross-Entropy, and Focal Loss to prioritize adenocarcinoma recall.

## Features

- **Custom Dataset**  
  - Organize your data under:
    ```
    data/
      lung_n/      # normal tissue
      lung_aca/    # adenocarcinoma
      lung_scc/    # squamous cell carcinoma
    ```
  - PyTorch `Dataset` & `DataLoader` for easy image loading and batching.

- **Transfer Learning**  
  - Pretrained ResNet-50 backbone (frozen layers 1–3).  
  - Trainable classification head:  
    ```
    Linear → ReLU → Dropout → Linear
    ```

- **Loss Functions**  
  - Weighted `CrossEntropyLoss`  
  - `FocalLoss` (γ = 2.0) for hard-example emphasis  
    > Lin et al., “Focal loss for dense object detection,” arXiv:1708.02002

- **Training & Evaluation**  
  - 70/15/15 train/validation/test split  
  - TQDM progress bars  
  - Confusion matrix & classification report after each epoch  

- **Interactive Visualization Dashboard**  
  - Built with [Dash](https://dash.plotly.com/), the app enables real-time model inspection:
    - 🎯 Select a ground-truth class
    - 🖼️ Sample an image from that class
    - 📊 View predicted softmax probabilities
    - 🔥 Overlay **Grad-CAM** heatmaps to see where the model is looking
    - 🧠 Visualize **Saliency maps** to reveal fine-grained pixel influence
  - Example output:
    <p align="center">
      <img src="docs/saliency.png" alt="Saliency Example" width="45%">
      <img src="docs/softmax.png" alt="Softmax Probabilities" width="45%">
    </p>

  - Launch locally:
    ```bash
    pip install dash plotly torch torchvision torchcam pillow
    python visualize.py
    ```

---

## Future Work

- **Two-stage classification pipeline**
  - Stage 1: Binary classifier for malignant vs. benign.
  - Stage 2: Subtype classifier distinguishing adenocarcinoma vs. SCC only on malignant cases.

- **Finetuning with Bayesian Optimization**

- **Drag-and-drop image analysis**
  - Allow custom images to be uploaded via dashboard

- **Quantitative explainability metrics**
  - Validate model focus with expert-annotated ROIs



