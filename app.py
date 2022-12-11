import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = load_model("Saved_models_malaria/model_vgg19.h5")
class_names = ['Parasitized', 'Uninfected']
input_size = [224, 224]

@app.get("/", response_class= HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    bytesImage = await file.read() # read file as bytes

    # Read the file (image) uploaded by the user
    image = read_image(bytesImage)

    # pre-process the image
    image = preprocess(image)

    # Make prediction
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions[0])] # Take a single image from the predicted batch and predict the class name using the index of the maximum argument
    confidence = round(100*np.max(predictions[0]), 2)

    #return {"Class": predicted_class,
    #       "Confidence": confidence}
    return f"{predicted_class} (Confidence: {confidence}%)"


def read_image(image):
    bytesImage = BytesIO(image)
    open_bytesImage = Image.open(bytesImage)
    return open_bytesImage

def preprocess(image: Image.Image):
    image = image.resize(input_size)
    image = np.asfarray(image) # convert into numpy array
    image = image/255 # rescale image
    image = np.expand_dims(image,0) # Transform to image batch by expanding the dimension because function predict only accept batch-images and not a single image
    return image


if __name__ == "__main__":   
    uvicorn.run(app, host = 'localhost', port = 8000, reload= True)
   