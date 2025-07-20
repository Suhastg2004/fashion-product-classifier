# Fashion Product Classifier

This repository contains a PyTorch-based multi-class fashion product classifier. It uses EfficientNet-B0 to classify fashion images into various categories such as article type, base colour, and gender.

## 📁 Project Structure

```
fashion_classifier/
├── api.py                  # FastAPI backend for prediction
├── app.py                  # Streamlit frontend application
├── model/                  # Directory containing trained model files
├── __pycache__/            # Python cache files
├── requirements.txt        # Python dependencies
├── fashion-product-classifier.ipynb  # Main training notebook
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/fashion_classifier.git
cd fashion_classifier

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate.bat  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## 🧠 Model Architecture

- **Base Model**: Pretrained `efficientnet_b0`
- **Multi-task Heads**:
  - Gender classification: 3 classes
  - BaseColour classification: 20 classes
  - ArticleType classification: 44 classes

All heads are trained jointly for simultaneous multi-label prediction.

## 📦 Model Training

Model training and evaluation are handled in the notebook:

```
fashion-product-classifier.ipynb
```

It includes:
- Label encoding for multi-task outputs
- Dataset preparation using PyTorch and PIL
- Training loop for multi-task learning
- Evaluation and accuracy/loss visualization

## 🚀 How to Run

### 1. Launch Streamlit App

```bash
streamlit run app.py
```

This opens a browser-based interface to upload images and view predictions.

### 2. Launch FastAPI Server (Optional)

```bash
uvicorn api:app --reload
```

This starts the FastAPI server for programmatic access to the model via HTTP requests.

## ✅ Sample Output

The app displays the predicted:
- Gender
- Base Colour
- Article Type

based on the uploaded image using the trained multi-head model.

## 📌 Dependencies

All dependencies are listed in `requirements.txt`. Key ones include:
- `torch`
- `torchvision`
- `streamlit`
- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pillow`
- `matplotlib`
- `pandas`

## 📄 License

This project is licensed under the MIT License.

---

👤 Developed by [Your Name]
