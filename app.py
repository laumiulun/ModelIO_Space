from tkinter.tix import COLUMN
import streamlit as st
import numpy as np
import pandas as pd
import PIL
import torch
import pickle
import boto3

# import time
import torchvision.transforms as transforms

class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()


def read_image_from_s3(bucket, key, region_name='us-west-2'):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3 = boto3.resource('s3', region_name=region_name)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream).convert('RGB')
    return im


def make_predictions():
    print("hi")

# ---- Title Screen -----------
st.title('Image Optimization: Email Industry')

# image = Image.Open('figures/ModelIO.png')

img = PIL.Image.open('figures/IO.png')
st.image(img)

st.markdown('Adding an image to an email campaign that will provide optimal engagement metrics can be challenging. How do you know which image to upload to your HTML, that will make an impact or significantly move the needle? And why would this image garner the best engagement? This model seeks to help campaign engineers understand which images affect their user engagement rate the most. The specific model is implemented using ResNet 18 and ResNet 34 for image embeddings extraction, and then we used these image embeddings as further inputs into a Gradient Boosted Tree model to generate probabilities on a user-specified target variable. The base model was adapted to car images and accurately predicted the user engagement rates with 91% accuracy. This model is adaptable for any large-scale marketing campaign using images. This model will identify the best images for optimal engagement for an email marketing campaign and serve engagement metrics prior to campaign launch. The model serves up several different images in milliseconds, so the campaign engineer understands which image to select in the campaign for optimized engagement.')

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    upload_img = PIL.Image.open(uploaded_file)
    st.image(upload_img, caption='Uploaded Image', width=300)
else:
    upload_img = None
    # st.write("")
    # st.write("Classifying...")
    # label = predict_label(image)
    # st.write('%s (%.2f%%)' % (label[0], label[1]*100))


# Drop down menu

target_variables = ['Open Rate',
                    'Click Through Open Rate',
                    'Revenue Generated per Email',
                    'Conversion Rate']
campaign_types = ['Abandoned Cart',
                  'Newsletter',
                  'Promotional',
                  'Survey',
                  'Transactional',
                  'Webinar',
                  'Engagement', 
                  'Review_Request', 
                  'Product_Announcement']

industry_types =['Energy',
                 'Entertainment',
                 'Finance and Banking',
                 'Healthcare',
                 'Hospitality',
                 'Real Estate', 'Retail', 'Software and Technology']


target = st.selectbox('Target Variables',target_variables, index=0)
campaign = st.selectbox('Campaign Types',campaign_types, index=0)
industry = st.selectbox('Industry Types',industry_types, index=0)


if st.button('Generate Predictions'):
    if upload_img is None:
        st.error('Please upload an image')
    else:
        placeholder = st.empty()
        placeholder.write("Loading Data...")

        # Starting Predictions

        data = pd.read_csv('data/wrangled_data_v2.csv', index_col=0)
        data_mod = data.copy()
        data_mod = data[(data.campain_type == campaign) & (data.industry == industry)]

        embeddings_df = pd.read_csv('data/embeddings_df.csv',index_col=0)
        embeddings_df = embeddings_df.iloc[data.index]


        # Transform to tensor 
        # transforming user input PIL Image to tensor

        # single_img_path = list(uploaded_image.value.keys())[0]
        single_image = upload_img.convert('RGB') # converting grayscale images to RGB
        # st.image(single_image, caption='Uploaded Image', width=300)

        my_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])

        image_tensor = my_transforms(single_image).unsqueeze(0) # transforming into tensor, unsqueeze to match input batch size dimensions


    
        placeholder.write('Loading Model...')

        model_path = 'model/my_checkpoint1.pth'
        model = torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        image_imbeddings = SaveFeatures(list(model._modules.items())[-1][1])

        with torch.no_grad():
            outputs = model(image_tensor)  # switched for cpu: image_tensor.cuda() (no cuda)
        img_embeddings = image_imbeddings.features[0]


        xgb_model = pickle.load(open("model/xgb_grid_model.pkl", "rb"))
        col_names = ['Abarth', 'Cab', 'Convertible', 'Coupe', 'GS', 'Hatchback', 'IPL', 'Minivan', 'R', 'SRT-8', 'SRT8', 'SS', 'SUV', 'Sedan', 'SuperCab', 'Superleggera', 'Type-S', 'Van', 'Wagon', 'XKR', 'Z06', 'ZR1']
        img_df = pd.DataFrame([img_embeddings], columns=col_names)

        #####
        # Getting Probabilities for Subsetted Dataframe
        full_df_probs = xgb_model.predict_proba(embeddings_df)
        full_df_probs = [i[1] for i in full_df_probs]
        prob_series = pd.Series(full_df_probs, index= embeddings_df.index)

        # 2 from each 
        top_10 = prob_series.sort_values(ascending=False)[:20]
        random_4_from_top_10 = top_10.sample(replace=False,n=2)
        
        # 2 from top 10 to 100
        top_10_100 = prob_series.sort_values(ascending=False)[20:100]
        random_4_from_top_10_100 = top_10_100.sample(replace=False,n=2)

        alternate_probs = pd.concat([random_4_from_top_10, random_4_from_top_10_100], axis=0)

        ######
        # Making predictions on user input and displaying results:
        img_pred = xgb_model.predict(img_df)[0]
        img_proba = xgb_model.predict_proba(img_df)[0][1]

        ######
        # making dictionary for max probability for recommendation
        max_prob_dict = {}
        max_prob_dict['current_image'] = img_proba
        for i in range(len(alternate_probs)):
            max_prob_dict['Alternate Image '+ str(i+1)] = alternate_probs.values[i]

        st.write('Below are the probabilities if alternate recommended images were used')

        img_index = alternate_probs.index[0]
        img_path = data.iloc[img_index][0]
        bucket = 'sagemaker-us-west-2-647020561811'
        key = 'sagemaker/Marlov-Image/'

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket)
        # for obj in bucket.objects.filter(Prefix=key):
        for obj in bucket.objects.all():
            key = obj.key
            body = obj.get()['Body'].read()
        # alt_img = read_image_from_s3(bucket,key,img_path)

        placeholder.empty()