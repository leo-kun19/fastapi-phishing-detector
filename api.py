from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import numpy as np
import os
import logging
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Inisialisasi FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ubah dengan asal tertentu jika perlu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi model dan scaler global
loaded_model = None
scaler = None

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the static directory path for serving HTML
static_dir = "static"

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))
    
# Definisikan format data input dengan Pydantic
class PhishingPredictionInput(BaseModel):
    url: HttpUrl  # Menggunakan HttpUrl untuk validasi URL

# Fungsi untuk mengekstrak fitur dari URL
def extract_features(url: str) -> dict:
    parsed_url = urlparse(url)
    
    features = {
        'NumDots': url.count('.'),
        'UrlLength': len(url),
        'NumDash': url.count('-'),
        'AtSymbol': url.count('@'),
        'IpAddress': int(any(char.isdigit() for char in parsed_url.netloc)),
        'HttpsInHostname': int(parsed_url.scheme == 'https'),
        'PathLevel': len(parsed_url.path.split('/')) - 1,
        'PathLength': len(parsed_url.path),
        'NumNumericChars': sum(c.isdigit() for c in url)
    }
    
    return features

@app.post("/upload_model/")
async def upload_model(my_model_file: UploadFile = File(...), scaler_file: UploadFile = File(...)):
    global loaded_model, scaler

    my_model_path = "temp_model.pkl"
    scaler_path = "temp_scaler.pkl"

    # Menyimpan file model dan scaler sementara
    with open(my_model_path, "wb") as f:
        f.write(await my_model_file.read())

    with open(scaler_path, "wb") as f:
        f.write(await scaler_file.read())

    # Memuat model dan scaler
    loaded_model = joblib.load(my_model_path)
    scaler = joblib.load(scaler_path)

    # Menghapus file sementara
    os.remove(my_model_path)
    os.remove(scaler_path)

    return {"message": "Model dan scaler berhasil diupload"}

@app.post("/predict_phishing/")
async def predict_phishing(phishing_input: PhishingPredictionInput):
    global loaded_model, scaler

    # Memastikan model dan scaler telah diupload
    if loaded_model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Model atau scaler belum diupload.")

    try:
        # Ekstrak fitur dari URL
        url_str = str(phishing_input.url)  # Pastikan untuk mengonversi ke string
        features = extract_features(url_str)
        
        logging.info(f"Extracted features: {features}")

        # Convert input menjadi DataFrame
        single_row_df = pd.DataFrame([features])

        # Kolom yang digunakan dalam prediksi
        X_columns = [
            'NumDots', 'UrlLength', 'NumDash', 'AtSymbol', 'IpAddress', 'HttpsInHostname', 'PathLevel',
            'PathLength', 'NumNumericChars'
        ]

        # Pastikan urutan kolom sesuai dan tambahkan kolom yang hilang
        for col in X_columns:
            if col not in single_row_df.columns:
                single_row_df[col] = 0

        single_row_df = single_row_df[X_columns]

        # Transformasi data
        single_row_scaled = scaler.transform(single_row_df)
        single_row_scaled = pd.DataFrame(single_row_scaled)

        # Prediksi dan probabilitas
        prediction = loaded_model.predict(single_row_scaled)
        prediction_proba = loaded_model.predict_proba(single_row_scaled)[:, 1]

        result = {
            "phishing_detection": "PHISHING" if prediction[0] == 1 else "NON PHISHING",
            "phishing_probability": f"{prediction_proba[0] * 100:.2f}%",
        }

        return result

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")
