import os
import cv2
import torch
import streamlit as st
import numpy as np
from src.eye_tracker import EyeTracker
from src.models.unet_baseline import UNet
from src.models.attention_unet import AttUNet
from src.grad_cam_proxy import GradCAM
from src.metrics import dice_coef, iou_score


# -------------------------------
# Utility: get model
# -------------------------------
def get_model(name, in_ch):
    if name == "unet":
        return UNet(in_ch=in_ch, out_ch=1, base=32)
    elif name == "attunet":
        return AttUNet(in_ch=in_ch, out_ch=1, base=32)
    else:
        raise ValueError("Unknown model type")


# -------------------------------
# Prediction + Grad-CAM
# -------------------------------
def predict_image(model_type, use_gaze, image, show_cam=False,
                  image_size=128, save_dir="saved_models"):
    device = torch.device("cpu")

    suffix = "gaze" if use_gaze else "baseline"
    auto_name = f"{model_type}_{suffix}_cpu.pth"
    weights_path = os.path.join(save_dir, auto_name)

    if not os.path.exists(weights_path):
        st.error(f"❌ Model file not found: {weights_path}")
        return None, None, None, None

    state = torch.load(weights_path, map_location="cpu")

    # Detect input channels from checkpoint
    first_conv_weight = state["inc.double_conv.0.weight"]
    ckpt_in_ch = first_conv_weight.shape[1]
    expected_in_ch = 2 if use_gaze else 1

    corrected = False
    if ckpt_in_ch != expected_in_ch:
        corrected = True
        use_gaze = (ckpt_in_ch == 2)

    model = get_model(model_type, ckpt_in_ch).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Preprocess uploaded image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig_shape = img.shape
    img_resized = cv2.resize(img, (image_size, image_size))
    img_norm = img_resized / 255.0
    img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 👀 Real gaze integration
    if use_gaze:
        tracker = EyeTracker()
        gaze_mask = tracker.get_gaze_mask(image_size=image_size)
        tracker.release()

        if gaze_mask is not None:
            gaze_tensor = torch.tensor(gaze_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            img_tensor = torch.cat([img_tensor, gaze_tensor], dim=1)
        else:
            st.warning("⚠️ Could not detect eyes, falling back to baseline segmentation.")

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs[0, 0] > 0.5).cpu().numpy().astype("uint8")

    pred_resized = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    heatmap = None
    if show_cam:
        target_layer = model.inc.double_conv[-1] if hasattr(model.inc, "double_conv") else model.inc
        gradcam = GradCAM(model, target_layer)
        heatmap_small = gradcam.generate(img_tensor)
        heatmap = cv2.resize(heatmap_small[0].cpu().numpy(),
                             (orig_shape[1], orig_shape[0]))

    return img, pred_resized, heatmap, {
        "weights_path": weights_path,
        "ckpt_in_ch": ckpt_in_ch,
        "corrected": corrected,
        "final_use_gaze": use_gaze,
        "model_type": model_type
    }


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🩺 AI-Assisted Ultrasound Segmentation")

st.sidebar.header("⚙️ Settings")

# Clinician-friendly labels
segmentation_method = st.sidebar.selectbox(
    "Segmentation Method",
    ["Standard Segmentation", "Enhanced Segmentation"]
)
model_type = "unet" if segmentation_method == "Standard Segmentation" else "attunet"

use_gaze = st.sidebar.checkbox("Focused-Area Assistance (Eye Gaze)", value=False)
show_cam = st.sidebar.checkbox("Show AI Attention Regions", value=False)

# Developer mode toggle
developer_mode = st.sidebar.checkbox("👨‍💻 Developer Mode", value=False)

uploaded_file = st.file_uploader("Upload an ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Uploaded Scan", use_container_width=True)

    if st.button("Run AI Analysis"):
        input_img, pred_mask, heatmap, info = predict_image(model_type, use_gaze, img, show_cam)

        if input_img is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(input_img, caption="Original Scan", use_container_width=True, channels="GRAY")
            with col2:
                st.image(pred_mask * 255, caption="AI Segmentation", use_container_width=True, channels="GRAY")
            if show_cam and heatmap is not None:
                with col3:
                    st.image(heatmap, caption="AI Attention Map", use_container_width=True)

            # ✅ Doctor view
            st.success("✅ AI analysis completed successfully.")

            # 👨‍💻 Developer details (hidden unless toggle is ON)
            if developer_mode and info is not None:
                st.markdown("---")
                st.subheader("🔍 Developer Details")
                st.write(f"**Requested model:** {info['model_type']}")
                st.write(f"**Weights file:** {info['weights_path']}")
                st.write(f"**Checkpoint expects input channels:** {info['ckpt_in_ch']}")
                st.write(f"**Focused-Area Assistance (Gaze):** {info['final_use_gaze']}")
                if info["corrected"]:
                    st.warning("⚠️ Input channels auto-corrected to match checkpoint.")

            # Optional Evaluation (if GT mask exists)
            dataset_root = "Dataset_BUSI_with_GT"
            file_name = uploaded_file.name
            possible_mask = None

            if "benign" in file_name:
                mask_path = os.path.join(dataset_root, "benign_mask", file_name.replace(".png", "_mask.png"))
                if os.path.exists(mask_path):
                    possible_mask = mask_path
            elif "malignant" in file_name:
                mask_path = os.path.join(dataset_root, "malignant_mask", file_name.replace(".png", "_mask.png"))
                if os.path.exists(mask_path):
                    possible_mask = mask_path
            elif "normal" in file_name:
                mask_path = os.path.join(dataset_root, "normal_mask", file_name.replace(".png", "_mask.png"))
                if os.path.exists(mask_path):
                    possible_mask = mask_path

            if possible_mask is not None:
                gt_mask = cv2.imread(possible_mask, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))
                gt_mask = (gt_mask > 127).astype("uint8")

                dice_val = dice_coef(torch.tensor(pred_mask), torch.tensor(gt_mask)).item()
                iou_val = iou_score(torch.tensor(pred_mask), torch.tensor(gt_mask)).item()

                st.success(f"✅ Agreement with Expert Annotation: Dice = {dice_val:.4f}, IoU = {iou_val:.4f}")
                st.image(gt_mask * 255, caption="Expert Annotation", use_container_width=True, channels="GRAY")
            else:
                st.info("ℹ️ No expert annotation found for this scan → metrics skipped.")
