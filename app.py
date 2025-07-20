import streamlit as st
from PIL import Image
import requests
import io

API_URL = "http://localhost:8000/predict/"

st.set_page_config(page_title="Fashion Product Classifier", layout="centered")
st.title("🛍️ Fashion Product Classifier")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg","png","jpeg"])
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare file for API
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        preds = response.json()
        st.markdown("**Predictions:**")
        st.write(f"**Colour:** {preds['colour']}")
        st.write(f"**Product Type:** {preds['product_type']}")
        st.write(f"**Season:** {preds['season']}")
        st.write(f"**Gender:** {preds['gender']}")
    else:
        st.error("Failed to get prediction from API.")