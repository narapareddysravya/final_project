import streamlit as st
import json
import os
from PIL import Image
import torch
import timm
from torchvision import transforms
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="centered")

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

# User management
def load_users():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    save_users(users)
    return True

def authenticate(username, password):
    users = load_users()
    return users.get(username) == password

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="Sravya-narapareddy/lung_cancer", filename="model1.pth")
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction logic
def predict_image(image):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_class = class_names[predicted.item()]
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
    return pred_class, confidence

# UI layout
st.markdown("<h1 style='text-align: center;'>🫁 Lung Cancer Detector</h1>", unsafe_allow_html=True)

# Sidebar
menu = ["Login", "Register", "Predict"]
choice = st.sidebar.selectbox("📋 Menu", menu)

if st.session_state.logged_in:
    st.sidebar.success(f"✅ Logged in as: {st.session_state.username}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

# Main sections
if choice == "Register":
    st.subheader("🔐 Create Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("✅ Registration successful. Please log in.")
        else:
            st.error("❌ Username already exists.")

elif choice == "Login":
    st.subheader("🔓 Login to Continue")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success(f"🎉 Welcome back, {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("❌ Invalid username or password.")

elif choice == "Predict":
    if st.session_state.logged_in:
        st.subheader("📷 Upload Chest X-ray")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            if uploaded_file.size > 2 * 1024 * 1024:  # 2MB limit
                st.error("⚠️ Image too large! Please upload an image smaller than 2MB.")
            else:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="📎 Uploaded Image", use_column_width=True)
                pred_class, confidence = predict_image(img)
                st.success(f"🩻 **Prediction**: `{pred_class}`")
                st.info(f"🔍 **Confidence**: `{confidence * 100:.2f}%`")
    else:
        st.warning("🔑 Please login to access prediction features.")

# Footer
st.markdown(
    """<hr style="border:1px solid #ddd"/>
    <p style="text-align:center;font-size:12px;">
    Developed with good will by <b>AIML</b> • Powered by Hugging Face & Streamlit
    </p>""",
    unsafe_allow_html=True
)
