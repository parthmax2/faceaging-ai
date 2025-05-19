
# FaceAging AI — Realistic Face Aging and De-Aging with AI

**FaceAging AI** is an advanced AI-powered web application that transforms face images to appear older or younger with realistic results. It leverages deep learning models for face detection and age transformation, offering an intuitive interface for users to upload images and see instant aged or de-aged outputs.

---

## Features

* **Face Aging & De-Aging**: Convert young faces to old and vice versa with high visual fidelity.
* **Automatic Face Detection**: Detects faces in uploaded images using OpenCV to process only valid faces.
* **Base64 Image Encoding**: Returns transformed images efficiently encoded for seamless frontend display.
* **FastAPI Backend**: Robust and scalable backend API handling image processing and AI inference.
* **Simple, Responsive UI**: User-friendly frontend using HTML, CSS, JavaScript, and Jinja2 templates.
* **CORS Enabled**: Allows cross-origin requests for flexible frontend-backend integration.

---

## Tech Stack

* **Backend**: FastAPI (Python)
* **Frontend**: HTML, CSS, JavaScript, Jinja2 Templates
* **AI & Image Processing**: OpenCV, Pillow, NumPy, Custom Face Aging Models
* **Deployment**: Cloud-ready (Render, Heroku, or any ASGI-compatible platform)

---

## Project Structure

```
faceaging-ai/
├── main.py                     # FastAPI app entry point with endpoints
├── helper.py                   # AI face aging helper functions & models
├── static/                     # Static files (CSS, JS, images)
├── templates/                  # HTML templates (Jinja2)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── Procfile                    # For deployment (if using Heroku/Render)
```

---

## Setup and Installation

### Prerequisites

* Python 3.8+
* Virtual environment tool (venv or conda)
* FastAPI, Uvicorn, OpenCV, Pillow, NumPy (see requirements.txt)

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/parthmax2/faceaging-ai.git
cd faceaging-ai
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the FastAPI development server**

```bash
uvicorn main:app --reload
```

5. **Access the app**

Open your browser and navigate to:

```
http://127.0.0.1:8000
```

Upload a face image, select the conversion type (Young to Old or Old to Young), and click Generate to see the transformed image.

---

## API Endpoints

* `GET /` — Serves the main web interface.
* `POST /convert/` — Accepts an image file and conversion type, returns the aged or de-aged image as a base64 string.

---

## Deployment

You can deploy this FastAPI app on any ASGI-compatible platform:

* **Render:** Easy cloud deployment with automatic Dockerfile or Python environment detection.
* **Heroku:** Use the provided `Procfile` and `requirements.txt`.
* **Other platforms:** Ensure support for Python 3.8+, ASGI, and WebSocket if needed.

---

## Contribution

Contributions are welcome! Please open issues or submit pull requests for:

* Improving model accuracy
* Enhancing UI/UX
* Adding new features or endpoints
* Optimizing performance

---

## License

MIT License — free to use, modify, and distribute.

---

## Contact

**Saksham Pathak**
Master’s in Artificial Intelligence & Machine Learning, IIIT Lucknow
[GitHub](https://github.com/parthmax2) | [LinkedIn](https://linkedin.com/in/sakshampathak) | [Instagram](https://instagram.com/parthmax_)

---

*FaceAging AI ©  Saksham Pathak. Powered by open-source AI and computer vision technologies.*

