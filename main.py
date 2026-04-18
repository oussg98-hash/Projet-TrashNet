from fastapi import FastAPI, File, UploadFile
import pickle
from PIL import Image
import numpy as np
import io

app = FastAPI()

with open("model_knn.pkl", "rb") as f:
    model = pickle.load(f)

categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@app.get("/")
def home():
    return {"message": "API fonctionne"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img) / 255.0
    img_flat = img_array.reshape(1, -1)

    prediction = model.predict(img_flat)[0]
    label = categories[prediction]

    return {"prediction": label}

#Le modèle KNN a été déployé à l’aide de FastAPI.
#Une API a été créée permettant d’envoyer une image et de recevoir la classe prédite en temps réel.