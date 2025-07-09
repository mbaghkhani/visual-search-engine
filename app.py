import streamlit as st
import torch
import numpy as np
from PIL import Image
import faiss
from transformers import CLIPProcessor, CLIPModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

index = faiss.read_index("clip_faiss_index.index")
paths = np.load("clip_image_paths.npy", allow_pickle=True)

def extract_query_feature(image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features[0].cpu().numpy().reshape(1, -1).astype("float32")
    faiss.normalize_L2(vec)
    return vec

st.title("Image Search Engine")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Input image", width=300)

    with st.spinner("Searching..."):
        query_vec = extract_query_feature(img)
        _, indices = index.search(query_vec, k=10)

    st.write("Top results:")
    cols = st.columns(5)
    for i, col in zip(indices[0], cols * 2):
        path = paths[i]
        result = Image.open(path)
        col.image(result, caption=os.path.basename(os.path.dirname(path)), use_column_width=True)
