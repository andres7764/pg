# Servidor server eye usando Flask
#import tensorflow as tf
# Importando librerias para la 
import numpy as np
import io
import base64
from PIL import Image
import keras
from keras import backend as k
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, request, jsonify

app = Flask(__name__)

def loadModel():
    #Variable global para manejar el modelo en todo el scope del servidor
    global model
    model = load_model('modelSaveEye.h5') # Carga el modelo entrenado
    print(model.predict) # Imprime la instancia del modelo creado.
    print("Modelo cargado")

# Función para procesar las imagenes que recibe del aplicativo
def processingImage(image, target_size):
    if image.mode != "RGB": # Convierte la imagen a RGB si no esta convertida
        image = image.convert("RGB")
    image = image.resize(target_size) # Setea tamaño a la imagen  
    image = img_to_array(image) # Convierte la imagen a un array procesable 
    image = np.expand_dims(image, axis=0)
    return image

print("Iniciando carga h5")
loadModel()

# Define ruta para acceder por http y se definen los métodos
@app.route('/predictImg', methods=["POST"])
def predictImg():
    data = request.get_json(force=True)
    dataImg = data['img'] # Recibe imagen en formato JSON 
    decodedImg = base64.b64decode(dataImg) 
    imgRaw = Image.open(io.BytesIO(decodedImg))
    processed_img = processingImage(imgRaw,target_size=(224, 224)) # Preprocesa imagen 
    prediction = model.predict(processed_img).tolist() # Realiza la predicción y retorna los valores en una lista.
    print(prediction)
#    response = { "a": "b" }
    response = {
        "result":prediction
    }
    # retorna al app la predicción
    return jsonify(response) 

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)