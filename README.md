# 🚗 Vehicle Detection & Traffic Counting System

## 📌 Introduction

This project focuses on building a **Deep Learning model** to:

* Detect vehicles in traffic (car, motorbike, bus, truck)
* Track moving objects in video
* Count the number of vehicles passing through a specific area

The system is designed for real-world applications such as:

* Traffic monitoring
* Smart city systems
* Congestion analysis

---

## 🧠 Technologies Used

* YOLO (Ultralytics)
* Python
* OpenCV
* DeepSORT (Object Tracking)
* NumPy, Pandas, Matplotlib

---

## 📂 Project Structure

```
.
├── data/               # Dataset (raw, processed, labels)
├── models/             # Trained models (not included)
├── notebooks/          # Jupyter notebooks for experiments
├── outputs/            # Output results (images, videos, logs)
├── src/                # Source code
│   ├── train.py        # Train model
│   ├── detect.py       # Vehicle detection
│   ├── tracking.py     # Object tracking
│   ├── count.py        # Vehicle counting
│   ├── config.py       # Configurations
│   └── utils.py        # Helper functions
├── main.py             # Run full pipeline
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Installation

### 1. Clone repository

```
git clone <your-repo-link>
cd vehicle-detection-project
```

### 2. Create virtual environment

```
python -m venv venv
```

Activate:

* Windows:

```
venv\Scripts\activate
```

* Linux/Mac:

```
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run detection & counting

```
python main.py
```

### Train model

```
python src/train.py
```

---

## 📊 Features

* 🚗 Vehicle detection using YOLO
* 🎯 Object tracking (avoid duplicate counting)
* 🔢 Traffic counting system
* 📹 Video processing with real-time visualization
* 📈 Traffic analysis (optional)

---

## 📁 Dataset

* Dataset is **not included** in this repository
* You can:

  * Add your own dataset into `data/`
  * Or use public datasets (COCO, traffic datasets)

---

## 🤝 Team Workflow

1. Clone the project
2. Create your own branch
3. Commit your changes
4. Push and create Pull Request

```
git checkout -b feature/your-feature
git add .
git commit -m "your message"
git push origin feature/your-feature
```

---

## ⚠️ Notes

* Do NOT upload large files such as:

  * `.pt` model files
  * large datasets
  * output videos/images
* Models will be:

  * downloaded automatically
  * or trained locally

---

## 🚀 Future Improvements

* Real-time traffic dashboard
* Heatmap visualization
* Traffic congestion detection
* Web interface (Streamlit)

---

## 📧 Contact

Project developed for Deep Learning course.
