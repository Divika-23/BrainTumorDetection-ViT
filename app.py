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

st.markdown("""
<style>
body {
    background-color: #0e1117;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    text-align: center;
    font-size: 60px;
    font-weight: 900;
    color: #00FFDD;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}
.section {
    background-color: #ffffff;
    padding: 25px;
    margin: 20px auto;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    color: black;
}
.menu-button {
    font-size: 20px;
    color: white;
    margin-bottom: 8px;
    padding: 8px;
}
hr {
    border: 0.5px solid #ccc;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)


menu_items = {
    "🔍 Why Do Brain Tumors Happen?": "They occur due to abnormal cell growth triggered by radiation, genetics, immune problems, or unknown causes.",
    "📌 What Are the Reasons?": "- Genetic mutations<br>- Radiation exposure<br>- Environmental toxins<br>- Family history<br>- Immune system disorders",
    "🚫 Things to Avoid": "- Radiation without medical reason<br>- Smoking and alcohol<br>- Skipping medical check-ups<br>- Poor sleep and junk food",
    "✅ Do's and ❌ Don'ts": "<b>Do's:</b><br>- Follow treatments<br>- Take medications<br>- Inform doctor of symptoms<br><br><b>Don'ts:</b><br>- Skip MRI scans<br>- Ignore symptoms<br>- Self-medicate",
    "💊 Medications": "- Anti-seizure drugs<br>- Chemotherapy<br>- Steroids<br>- Pain control<br><i>Follow only your doctor’s prescription.</i>",
    "💡 Health Tips": "- Eat fruits/veggies<br>- Sleep 7–8 hours<br>- Stay hydrated<br>- Meditate<br>- Exercise"
}

if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = None


col1, col2, col3 = st.columns([1, 2.5, 0.5])

with col1:
    st.markdown("### 🧭 Info Menu")
    for item in menu_items:
        if st.button(item, key=item):
            st.session_state.selected_menu = item
        st.markdown("<hr>", unsafe_allow_html=True)

with col2:
    if st.session_state.selected_menu:
        selected = st.session_state.selected_menu
        st.markdown(f"""
        <div class="section">
            <h3>{selected}</h3>
            <p>{menu_items[selected]}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
        <h3>🧠 What is a Brain Tumor?</h3>
        <p>A brain tumor is an abnormal growth of cells in or around the brain...</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
        <h4>📤 Upload MRI Image</h4>
        <p>Drag and drop or browse a brain MRI image file (JPG, PNG).</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing image..."):
        time.sleep(2)

        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).item()

        classes = ["No Tumor", "Tumor Detected"]
        confidence = probs[0][preds].item() * 100

img_col, result_col = st.columns([2, 1], gap="small")

with img_col:
    st.image(image, caption="🖼️ Uploaded MRI Image", width=300)

with result_col:
    if preds == 1:
        st.markdown("""
        <div style="background-color:#ff4d4f; color:white; padding:18px; border-radius:12px;">
            <h3>🚨 Tumor Detected</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#52c41a; color:white; padding:18px; border-radius:12px;">
            <h3>✅ No Tumor Detected</h3>
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    Streamlit UI for Brain Tumor Detection using Vision Transformers
</div>
""", unsafe_allow_html=True)




















