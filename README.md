
# Visual Search Engine (Advanced Computer Vision Final Project)

This project implements a visual image retrieval system, where users can upload a photo and retrieve the most visually similar images from a local dataset using deep visual features and efficient indexing.

It was developed as the final project for the **Advanced Computer Vision** course (Filoger), Summer 2025.

---

## Overview

- Uses the pre-trained `openai/clip-vit-base-patch32` model to extract semantic image embeddings.
- Builds a FAISS index on top of the dataset features for fast similarity search.
- A Streamlit-based interface enables users to upload query images and see the most visually similar results.

---

##  Folder & File Structure

- `tiny_imagenet/` : Dataset folder (contains `train/` and `val/` subfolders).
- `app.py` : Streamlit web app for searching similar images.
- `visual-search-engine.ipynb` : Notebook to extract features and build the FAISS index.
- `clip_faiss_index.index` : FAISS index file (not included here due to size).
- `clip_image_paths.npy` : Array of image paths corresponding to vectors in the index (not included).
- `image_features_clip.npz` : Optional compressed file of extracted features (not included in repo).

> **Note**: Due to large file sizes, only core code files are uploaded. To fully run the app, please run the notebook to generate the missing files.

---

##  How to Use

### 1. Install Requirements

Install necessary Python packages:
```bash
pip install torch torchvision transformers streamlit faiss-cpu datasets pillow
```

### 2. Rebuild the Feature Index

Run the following notebook to extract features and build the FAISS index:
```bash
visual-search-engine.ipynb
```

This will generate:
- `clip_faiss_index.index`
- `clip_image_paths.npy`

### 3. Launch the App

Once the index is built, run:
```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

---

##  Notes

- Dataset used: [Tiny ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet) (via Hugging Face Datasets).
- Image embeddings are extracted using CLIP.
- Search is based on cosine similarity using normalized FAISS indexing.

---

##  Developed by

Summer 2025  
Advanced Computer Vision Course – Filoger Academy
