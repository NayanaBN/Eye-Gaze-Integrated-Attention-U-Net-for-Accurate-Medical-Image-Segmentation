
# **Eye-Gaze Integrated Attention U-Net for Accurate Medical Image Segmentation**

## **1. Introduction**

This work presents an **Eye-Gaze Integrated Attention U-Net**, a lightweight deep-learning framework designed to improve medical image segmentation accuracy and interpretability. The model is demonstrated on **breast ultrasound (BUS)** images and introduces gaze-guided attention as an additional modality to guide the segmentation network toward clinically significant regions.

The system aims to enhance diagnostic support in low-resource settings by maintaining CPU-only compatibility, lightweight architecture, and strong interpretability features.

---

## **2. Key Contributions**

### **2.1 Gaze-Guided Segmentation**

The framework incorporates human visual attention cues—represented as gaze maps—directly into the Attention U-Net architecture. This encourages the network to focus on diagnostically relevant areas, improving lesion localization and segmentation accuracy.

### **2.2 Hardware-Agnostic Gaze Proxy**

To avoid dependence on specialized eye-tracking hardware, the method employs a **Grad-CAM–based proxy gaze map** that simulates expert attention. The system is also compatible with gaze data captured using a regular webcam, enabling wider accessibility.

### **2.3 Two-Channel Input Fusion**

The model uses a two-channel input tensor that combines the raw ultrasound image with its corresponding gaze map. This allows the network to jointly process visual features and human-attention cues, leading to more focused and reliable segmentation outputs.

### **2.4 Lightweight and Deployable Architecture**

The entire pipeline is optimized for **CPU-only inference**, making it suitable for deployment in real-world clinical environments with limited computing resources.

### **2.5 Built-in Interpretability**

A Grad-CAM heatmap overlay is integrated into the deployment interface, providing clinicians with visual explanations of the model’s decision-making process and increasing trust in the system’s outputs.

---

## **3. Model Architecture**

The proposed system extends the **Attention U-Net** architecture by integrating human attention cues into the feature-selection process.

### **3.1 Input Design**

* **Input Channel 1:** Raw ultrasound image
* **Input Channel 2:** Corresponding gaze map (actual or proxy)

These two channels are concatenated into a single tensor before being passed through the network.

### **3.2 Encoder–Decoder Framework**

The model follows a traditional U-Net structure:

* **Encoder Path:** Sequential downsampling layers extract hierarchical features from the input.
* **Decoder Path:** Upsampling layers reconstruct the final segmentation mask.
* **Attention Gates:** Positioned at skip connections, these gates leverage the gaze map to emphasize features located in clinically important areas while suppressing irrelevant regions.

### **3.3 Output**

The system produces:

1. A binary segmentation mask
2. A Grad-CAM interpretability heatmap

These outputs are displayed through a Streamlit-based interface for easy clinical use.

---

## **4. Performance Evaluation**

The proposed model was tested on the **BUSI (Breast Ultrasound Images) dataset**. Performance was compared with standard U-Net and Attention U-Net baselines.

| **Model**                                  | **Dice Coefficient** | **Intersection over Union (IoU)** |
| ------------------------------------------ | -------------------- | --------------------------------- |
| U-Net (Baseline)                           | 0.412                | 0.297                             |
| Attention U-Net (Baseline)                 | 0.428                | 0.311                             |
| **Gaze-Guided Attention U-Net (Proposed)** | **0.4488**           | **0.3231**                        |

The proposed method demonstrated superior performance across both metrics.
The system also achieved significantly higher scores when using optimized proxy-gaze methods (Dice ≈ 0.912, IoU ≈ 0.845), highlighting the effectiveness of integrating gaze-based attention.

---

## **5. System Implementation and Deployment**

### **5.1 Environment**

* Programming Language: Python 3.8
* Framework: PyTorch
* Interface: Streamlit

The system is validated on a conventional Intel Core i5 CPU with 8 GB RAM, confirming that GPU acceleration is not required.

### **5.2 Inference Workflow**

1. The ultrasound image is preprocessed.
2. A gaze map is generated using the Grad-CAM–based proxy module or collected through a webcam.
3. The image and gaze map are concatenated as a two-channel tensor.
4. The trained model performs segmentation.
5. Grad-CAM overlays are generated for interpretability.
6. Both the segmentation mask and heatmap are displayed on the Streamlit interface.

---

## **6. Future Work**

Potential extensions of this research include:

* Incorporating temporal gaze patterns for video-based medical imaging.
* Testing on multi-center datasets and additional imaging modalities.
* Leveraging active learning or reinforcement learning to refine the proxy-gaze generator.
* Enhancing real-time interaction and enabling privacy-preserving training using federated learning.

---
