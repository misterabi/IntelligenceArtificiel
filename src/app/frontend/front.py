import io
import json
import cv2
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas


# Titre de l'application
st.title("Interface pour l'inférence d'image")

# Premier box : Zone de dessin à la main
st.subheader("Dessinez votre image ici :")
    

# Deuxième box : Visualisation de l'image sélectionnée
st.subheader("Visualisation de l'image :")
width_canvas = 200
height_canvas = 200
st.write("Drawn image")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=5,
    stroke_color="white",
    width=width_canvas,
    height=height_canvas,
    drawing_mode="freedraw",
    key="canvas",
    display_toolbar=True,
)

#Get Image

if canvas_result.image_data is not None:
    #scale down image to the model input size
    img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))  
    #rescale image to show
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rescaled = cv2.resize(img, (width_canvas, height_canvas))
    st.write("model input")
    st.image(img_rescaled)

# Predict Button


if st.button('Prédire l\'image'):

        # Envoi de l'image à l'API via FastAPI (exemple)
        api_endpoint = 'http://backend:8000/api/v1/predict'  # Remplacez par votre URL de l'API de prédiction
        print(img)
        image_pil = Image.fromarray(img)
        img_byte_arr = io.BytesIO()
        img_byte_arr = img_byte_arr.getvalue()
        print(img_byte_arr)
        response = requests.post(api_endpoint, data=json.dumps({'image': img.tolist()}))

        # Affichage du résultat de l'inférence
        if response.status_code == 200:
            result = response.json()
            st.success('Résultat de l\'inférence : {}'.format(result))
        else:
            st.error('Erreur lors de l\'envoi de l\'image à l\'API')