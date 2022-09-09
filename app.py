import streamlit as st
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision.transforms as transforms

import pickle

# AWS 
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config

# Plotly and Bokeh
import plotly.graph_objects as go
from bokeh.models.widgets import Div


def convert_percentage(score):
    rounded_probability = str(np.round(score*100,2)) + "%"
    return rounded_probability

def url_button(button_name,url):
    if st.button(button_name):
        js = """window.open('{url}')""".format(url=url) # New tab or window
        html = '<img src onerror="{}">'.format(js)
        div = Div(text=html)
        st.bokeh_chart(div)

def table_data():
    # creating table data
    field = [
        'Data Scientist',
        'Dataset',
        'Algorithm',
        'Framework',
        'Ensemble',
        'Domain',
        'Model Size'
    ]

    data = [
        'Andy Lau',
        'Stanford Cars Dataset',
        'Deep Learning Convolutional Neural Network: ResNet50',
        'Pytorch',
        'XGBoost',
        'ResNet Image Classification',
        '76.55 KB'
    ]

    data = {
        'Field':field,
        'Data':data
    }

    df = pd.DataFrame.from_dict(data)

    return df


def create_box(text,label):
    st.markdown(f'<p style="background-color:#d2e4f6;padding: 5px 5px;border-radius:10px;font-size:24px;"><center><b>{text}</b>: {label}</center></p>', unsafe_allow_html=True)

def create_table():
    # creating table data
    field = [
        'Data Scientist',
        'Dataset',
        'Algorithm',
        'Framework',
        'Ensemble',
        'Domain',
        'Model Size'
    ]

    data = [
        'Andy Lau',
        'Stanford Cars Dataset',
        'Deep Learning Convolutional Neural Network: ResNet50',
        'Pytorch',
        'XGBoost',
        'ResNet Image Classification',
        '76.55 KB'
    ]

    data = {
        'Field':field,
        'Data':data
    }

    df = pd.DataFrame.from_dict(data)


    header_color = ['#0f4d60','#1c8d99']
    cell_color = ['rgba(15,77,96,0.25)','rgba(28,141,153,0.33)']

    # Create figures  
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color=header_color,
                    font=dict(color='white', size=15),
                    align='left'),
        cells=dict(values=[df.Field, df.Data],
                fill_color=header_color,
                font=dict(color='white', size=15),
                align='left'))
    ])
    # Make the header dissapear 
    fig.for_each_trace(lambda t: t.update(header_fill_color = 'rgba(0,0,0,0)'))

    return fig


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


def read_image_from_s3(bucket, key):
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
    s3 = boto3.resource('s3',config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = PIL.Image.open(file_stream).convert('RGB')
    return im



# ---- Title Screen -----------
st.markdown('# Image Optimization: Email Industry')

# image = Image.Open('figures/ModelIO.png')

col1, col2, col3 = st.columns([1,1,1])

with col2:
    img = PIL.Image.open('figures/IO.png')
    st.image(img)
# with col2:
    # html3 = f"""
    #         <div class="total-dc"">
    #             <p>Total DC: £<p>
    #             <p>TEST<p>
    #         </div>

    #         """
    # st.markdown(html3, unsafe_allow_html=True)
    # st.markdown('#### Data Scientist')

stats_col1, stats_col2, stats_col3, stats_col4 = st.columns([1,1,1,1])

# with stats_col1:
#     # st.markdown(' **Production**: Ready',unsafe_allow_html=True)
#     create_box('Production','Ready')
# with stats_col2:
#     create_box('Accuracy','91%')
# with stats_col3:
#     create_box('Speed','2.18 ms')
# with stats_col4:
#     # st.markdown(' **Industry**: Email Marketing')
#     create_box('Industry','Email Marketing')

# st.markdown("""
# <style>
# div[data-testid="metric-container"] {
#    background-color: rgba(28, 131, 225, 0.1);
#    border: 1px solid rgba(28, 131, 225, 0.1);
#    padding: 5% 5% 5% 10%;
#    border-radius: 5px;
#    color: rgb(30, 103, 119);
#    overflow-wrap: break-word;
# }

# /* breakline for metric text         */
# div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
#    overflow-wrap: break-word;
#    white-space: break-spaces;
#    color: red;
# }
# </style>
# """
# , unsafe_allow_html=True)
with stats_col1:
    st.metric(label="Production", value="Ready")
with stats_col2:
    st.metric(label="Accuracy", value="91%")

with stats_col3:
    st.metric(label="Speed", value="2.18 ms")

with stats_col4:
    st.metric(label="Industry", value="Email")


# ---- Model Information -----------
# info_col1, info_col2, info_col3 = st.columns([1,1,1])
with st.sidebar:
    with st.expander('Model Description', expanded=False):
        st.markdown('Adding an image to an email campaign that will provide optimal engagement metrics can be challenging. How do you know which image to upload to your HTML, that will make an impact or significantly move the needle? And why would this image garner the best engagement? This model seeks to help campaign engineers understand which images affect their user engagement rate the most. The specific model is implemented using ResNet 18 and ResNet 34 for image embeddings extraction, and then we used these image embeddings as further inputs into a Gradient Boosted Tree model to generate probabilities on a user-specified target variable. The base model was adapted to car images and accurately predicted the user engagement rates with 91% accuracy. This model is adaptable for any large-scale marketing campaign using images. This model will identify the best images for optimal engagement for an email marketing campaign and serve engagement metrics prior to campaign launch. The model serves up several different images in milliseconds, so the campaign engineer understands which image to select in the campaign for optimized engagement.')

    with st.expander('Model Information', expanded=False):
        st.table(table_data())

    url_button('Model Homepage','https://www.loxz.com/#/models/IO')
    url_button('Full Report','https://resources.loxz.com/reports/image-optimization-model')
    url_button('Amazon Market Place','https://aws.amazon.com/marketplace')




uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    upload_img = PIL.Image.open(uploaded_file)
else:
    upload_img = None


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
        random_4_from_top_10 = top_10.sample(replace=False,n=1)
        
        # 2 from top 10 to 100
        top_10_100 = prob_series.sort_values(ascending=False)[20:100]
        random_4_from_top_10_100 = top_10_100.sample(replace=False,n=1)

        alternate_probs = pd.concat([random_4_from_top_10, random_4_from_top_10_100], axis=0)

        ######
        # Making predictions on user input and displaying results:
        img_pred = xgb_model.predict(img_df)[0]
        img_proba = xgb_model.predict_proba(img_df)[0][1]
        max_prob_dict = {}
        max_prob_dict['current_image'] = img_proba
        for i in range(len(alternate_probs)):
            max_prob_dict['Alternate Image '+ str(i+1)] = alternate_probs.values[i]

        st.write('Below are the probabilities if alternate recommended images were used')

        st.subheader('Original Image Probability')
        st.image(upload_img,caption = convert_percentage(img_proba),width=300)


        img_index_1 = alternate_probs.index[0]
        img_path_1 = data.iloc[img_index_1][0]

        img_index_2 = alternate_probs.index[1]
        img_path_2 = data.iloc[img_index_2][0]

        bucket = 'lozx-public-data'
        file_base = 'Model-IO/'
        im_1 = read_image_from_s3(bucket, file_base + img_path_1)
        im_2 = read_image_from_s3(bucket, file_base + img_path_2)


        alt_col1, alt_col2 = st.columns([1,1])
        with alt_col1:
            st.subheader("Alternate Image 1")
            st.image(im_1, caption=convert_percentage(alternate_probs.values[0]),width=300);
        with alt_col2:
            st.subheader("Alternate Image 2")
            st.image(im_2, caption=convert_percentage(alternate_probs.values[1]), width=300);


        placeholder.empty()