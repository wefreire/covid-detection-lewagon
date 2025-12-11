from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from model_manager import ModelManager
import configs
import utils

app = FastAPI(title="Simple Image Prediction API")

def load_image(file: UploadFile):
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Could not process image: {e}")


# @app.post("/predict_pca")
# async def predict_pca(image: UploadFile = File(...)):
#     img = load_image(image)
#     # Dummy output (replace with real PCA model)
#     # mm = ModelManager(configs.)

#     return JSONResponse({"model": "PCA", "prediction": "placeholder_result"})


# @app.post("/predict_cnn")
# async def predict_cnn(image: UploadFile = File(...)):
#     img = load_image(image)
#     # Dummy output (replace with real CNN model)
#     mm = ModelManager(configs.CNN_PATH)
#     prediction = mm.predict(img)
    
#     confidence = float(prediction[0][0])
#     predicted = utils.evaluate_confidence(confidence)
        
#     return predicted


@app.post("/predict_densenet121")
async def predict_densenet121(image: UploadFile = File(...)):
    img = load_image(image)
    breakpoint()
    mm = ModelManager(configs.DENSE_NET_PATH)
    breakpoint()
    prediction = mm.predict(img)
    breakpoint()
    confidence = float(prediction[0][0])
    predicted = utils.evaluate_confidence(confidence)
        
    return predicted


