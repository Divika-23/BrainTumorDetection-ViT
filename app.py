import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import time


@st.cache_resource
def load_model_and_extractor():
    num_classes = 2  # No Tumor / Tumor
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load("brain.pth", map_location=torch.device("cpu")))
    model.eval()

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    return model, feature_extractor

model, feature_extractor = load_model_and_extractor()

# --------------------------
# 2. Streamlit Page Setup
# --------------------------
st.set_page_config(page_title="Brain Tumor Detection | ViT", layout="wide", page_icon="🧠")




