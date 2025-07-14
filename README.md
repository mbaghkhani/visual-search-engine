
# Visual Search Engine (Advanced Computer Vision Final Project)

This project implements a visual image retrieval system, where users can upload a photo and retrieve the most visually similar images from a local dataset using deep visual features and efficient indexing.

It is developed as the final project for the Advanced Computer Vision course (Filoger).

---

## Overview

- The system uses a pre-trained CLIP model to extract semantic features from images.
- A FAISS index is built on all dataset features for fast similarity search.
- A web-based interface using Streamlit allows users to upload a query image and see similar results.

---

## Folder and File Structure

- `tiny_imagenet/` : Folder containing the image dataset, separated into `train/` and `val/`.
- `app.py` : Streamlit interface to run the visual search engine.
- `clip_faiss_index.index` : Pre-built FAISS index containing vector embeddings of all dataset images.
- `clip_image_paths.npy` : Numpy array of file paths corresponding to each vector in the index.
- `image_features_clip.npz` : Compressed file containing all extracted features and paths.
- `visual-search-engine.ipynb` : Notebook to extract features from dataset images and build the FAISS index.
> **Note**: Due to large file sizes, only core code files are uploaded. To fully run the app, please run the notebook to generate the missing files.

---

## How to Use

### 1. Install Requirements
Make sure to install the necessary Python libraries:
```bash
pip install torch torchvision transformers streamlit faiss-cpu datasets pillow
```

### 2. Preprocess the Dataset (optional)
If you need to extract features again, open and run `visual-search-engine.ipynb`.

### 3. Launch the App
In the terminal, run:
```bash
streamlit run app.py
```

Then open the link shown in your browser.

---

## Notes

- The project uses `openai/clip-vit-base-patch32` as the feature extractor.
- Dataset used: Tiny-ImageNet (downloaded using Hugging Face Datasets). https://huggingface.co/datasets/zh-plus/tiny-imagenet
- Similarity search is based on cosine similarity via normalized vectors in FAISS.

---

## Developed by
Summer 2025  
For Advanced Computer Vision Course (Filoger)
