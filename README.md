# VanCorp Holdings

# AI Virtual Try-On System

A Modular, Scalable, Production-Ready Virtual Try-On System powered by Deep Learning, designed to enable users to try clothes virtually using just an image. Powered by DL & vision.

          +----------------+       +------------------+        +-----------------+
          |   User Upload  | --->  |  Pose Estimation  |  --->  | Human Parsing   |
          +----------------+       +------------------+        +-----------------+
                   |                         |                          |
                   |                         v                          v
                   |            +-----------------------+     +------------------+
                   |            | Cloth Warping (TPS)   | <-- | Clothing Upload  |
                   |            +-----------------------+     +------------------+
                   |                         |
                   v                         v
          +------------------------------------------+
          |        Fusion & Rendering Module         |
          +------------------------------------------+
                              |
                              v
               +-------------------------------+
               |  Output Final Try-On Result   |
               +-------------------------------+

# ⚙️ Tech Stack
Component              	Technology Used
Framework             	FastAPI (Python)
Pose Estimation	        MediaPipe Pose / HRNet / OpenPose
Cloth Segmentation	    OpenCV (baseline), CIHP Parsing, DeepLabV3+
Warping Engine	        Thin Plate Spline (TPS), CP-VTON, VITON-HD
Fusion Generator	      TryOnGAN, SPADE, Diffusers (Stable Diffusion)
Image I/O	              OpenCV, NumPy
Real-time Support	      ONNX, TensorFlow Lite, MediaPipe

# 🚀 Features
✅ Upload body and cloth images
✅ Estimate body landmarks
✅ Segment and isolate clothing regions
✅ Wrap clothes using keypoints (TPS)
✅ Fuse cloth realistically using GAN or Diffusion
✅ Save and serve final rendered try-on result
✅ Easily extendable and modular

# Project Structure

google_tryon/
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── upload.py
│   │   └── tryon.py
│   ├── ml/
│   │   ├── pose_estimator.py
│   │   ├── cloth_segmentation.py
│   │   ├── cloth_warping.py
│   │   ├── fusion.py
│   │   └── tryon_pipeline.py
│   └── utils/
│       └── image_io.py
├── static/
│   ├── uploads/
│   │   ├── users/
│   │   └── cloths/
│   └── outputs/
├── models/
│   ├── cihp/
│   ├── viton/
│   └── tryongan/
├── requirements.txt
└── README.md

# 📦 Setup Instructions

1️⃣ Clone the Repository

git clone https://github.com/your-username/google-tryon.git
cd "your file name" 

2️⃣ Create Environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt
⚠️ Make sure you use Python 3.10 for MediaPipe & DL compatibility.

4️⃣ Start the FastAPI Server
uvicorn app.main:app --reload

Access Swagger UI at: http://localhost:8000/docs

# 🖥️ Minimum System Requirements

CPU: Intel i7/i9 (8+ cores)
GPU: NVIDIA RTX 3060/4060 or higher
RAM: 16GB+
Disk: 10GB+ free (for model weights)
Python: 3.10.x

# Contributors
~ Satyabrat Sahu
