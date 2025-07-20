from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uvicorn
import torch
import torchvision.transforms as T
import onnxruntime as ort
import numpy as np
import joblib

# Load label encoders
le_colour      = joblib.load("le_colour.pkl")
le_product     = joblib.load("le_product_type.pkl")
le_season      = joblib.load("le_season.pkl")
le_gender      = joblib.load("le_gender.pkl")

# Load ONNX model session
ort_session = ort.InferenceSession("model.onnx")

# Image transform
IMG_SIZE = 224
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

app = FastAPI(title="Fashion Classifier API")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    x = transform(img).unsqueeze(0).numpy()

    # Run ONNX inference
    inputs = {ort_session.get_inputs()[0].name: x}
    outputs = ort_session.run(None, inputs)
    colour_out, prod_out, season_out, gender_out = outputs

    # Decode predictions
    colour_pred  = le_colour.inverse_transform([np.argmax(colour_out, 1)[0]])[0]
    product_pred = le_product.inverse_transform([np.argmax(prod_out, 1)[0]])[0]
    season_pred  = le_season.inverse_transform([np.argmax(season_out, 1)[0]])[0]
    gender_pred  = le_gender.inverse_transform([np.argmax(gender_out, 1)[0]])[0]

    return JSONResponse({
        "colour": colour_pred,
        "product_type": product_pred,
        "season": season_pred,
        "gender": gender_pred
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)