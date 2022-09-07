import streamlit as st
import numpy as np
import pandas as pd
import PIL

# ---- Title Screen -----------
st.title('Image Optimization: Email Industry')

# image = Image.Open('figures/ModelIO.png')

img = PIL.Image.open('figures/IO.png')
st.image(img)

st.markdown('Adding an image to an email campaign that will provide optimal engagement metrics can be challenging. How do you know which image to upload to your HTML, that will make an impact or significantly move the needle? And why would this image garner the best engagement? This model seeks to help campaign engineers understand which images affect their user engagement rate the most. The specific model is implemented using ResNet 18 and ResNet 34 for image embeddings extraction, and then we used these image embeddings as further inputs into a Gradient Boosted Tree model to generate probabilities on a user-specified target variable. The base model was adapted to car images and accurately predicted the user engagement rates with 91% accuracy. This model is adaptable for any large-scale marketing campaign using images. This model will identify the best images for optimal engagement for an email marketing campaign and serve engagement metrics prior to campaign launch. The model serves up several different images in milliseconds, so the campaign engineer understands which image to select in the campaign for optimized engagement.')





#