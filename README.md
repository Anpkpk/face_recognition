# Face Recognition Project

A simple face recognition application built with Python.  
Includes a GUI and a core face recognition engine.

## 🧠 Model

The core of this project is a **Siamese Neural Network** trained with contrastive loss.  
It works by comparing two face embeddings and measuring their similarity,  
instead of directly classifying faces.  

Advantages:
- Handles unseen faces better than closed-set classifiers
- Works well with limited data
- Embedding-based, so scalable for large face databases


---

## 🚀 Requirements
- Python > 3.8, < 3.12
- pip or conda for dependency management

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Anpkpk/face_recognition.git
cd face_recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## ▶️ Usage

Run the main application:

```bash
python -m src.main
```


