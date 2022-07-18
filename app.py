import streamlit as st
import cv2
import numpy as np
import time as t
from tensorflow.keras.models import load_model

st.set_page_config(page_title=None, page_icon=None, layout='wide', initial_sidebar_state='collapsed')
html_temp = """
    <div style="position: relative; width: 100%; height: 0; padding-top: 48.1481%;
 padding-bottom: 48px; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
 border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAEhJwlb2KA&#x2F;view?embed">
  </iframe>
</div>
    """
st.markdown(html_temp, unsafe_allow_html = True)

def image_read(image):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    cv2.resize(img, (224, 224))

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    im_color = cv2.resize(im_color, (224, 224))

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (224, 224))
    img1 = np.array(img1) / 255
    img1 = np.expand_dims(img1, axis=0)

    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    ret,thresh = cv2.threshold(gray_image,100,300,0) 
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    im_with_contours = cv2.drawContours(img,contours,-1,(0,300,0),1)
    im_with_contours = cv2.resize(im_with_contours, (224, 224))

    return img1, im_color, im_with_contours

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def model_load(sel2,img1):
    if sel2 == 'CT-scan':
            c2_model = load_model("inception_ct.h5")
            c3_model = load_model("vgg_ct.h5")
            image1 = cv2.imread("inception_ct_report.png")
            image2 = cv2.imread("vgg_ct_report.png")
            with st.spinner(text="Downloading model weights... this may take up to a few minutes. (~150 MB) Please don't interrupt it."):
                t.sleep(10)
            c2_pred = c2_model.predict(img1)
            c3_pred = c3_model.predict(img1)
            p_c2 = c2_pred[0]
            p_c3 = c3_pred[0]
            if p_c2[0] > 0.5:
                r_c2 = str('%.2f' % (p_c2[0] * 100) + '% COVID POSITIVE')
            else:
                r_c2 = str('%.2f' % ((1 - p_c2[0]) * 100) + '% COVID NEGATIVE')
            if p_c3[0] > 0.5:
                r_c3 = str('%.2f' % (p_c3[0] * 100) + '% COVID POSITIVE')
            else:
                r_c3 = str('%.2f' % ((1 - p_c3[0]) * 100) + '% COVID NEGATIVE')

            st.write("prediction table")
            cols = st.beta_columns(2)
            cols[0].write('InceptionV3')
            cols[0].image(image1, use_column_width=True, caption=r_c2)
            cols[1].write('Xception')
            cols[1].image(image2, use_column_width=True, caption=r_c3)
    elif sel2 == 'X-ray':
            c4_model = load_model("inception_chest.h5")
            c5_model = load_model("vgg_chest.h5")
            image1 = cv2.imread("inception_chest_report.png")
            image2 = cv2.imread("vgg_chest_report.png")
            with st.spinner(text="Downloading model weights... this may take up to a few minutes. (~150 MB) Please don't interrupt it."):
                t.sleep(10)
            c4_pred = c4_model.predict(img1)
            c5_pred = c5_model.predict(img1)
            p_c4 = c4_pred[0]
            p_c5 = c5_pred[0]
            if p_c4[0] > 0.5:
                    r_c4 = str('%.2f' % (p_c4[0] * 100) + '% COVID POSITIVE')
            else:
                r_c4 = str('%.2f' % ((1 - p_c4[0]) * 100) + '% COVID NEGATIVE')

            if p_c5[0] > 0.5:
                    r_c5 = str('%.2f' % (p_c5[0] * 100) + '% COVID POSITIVE')
            else:
                r_c5 = str('%.2f' % ((1 - p_c5[0]) * 100) + '% COVID NEGATIVE')
            st.write("prediction table")
            cols = st.beta_columns(2)
            cols[0].write('InceptionV3')
            cols[0].write(r_c4)
            cols[0].image(image1, use_column_width=True)
            cols[1].write('VGG16')
            cols[1].write(r_c5)
            cols[1].image(image2, use_column_width=True)

    st.success('Done!')
    st.balloons()

uploaded_image = st.file_uploader('select')
sel2 = st.selectbox('', ['Select the mode of Diagnosis', 'CT-scan','X-ray'])
    
if uploaded_image is not None and (sel2 == 'CT-scan' or sel2 == 'X-ray'):
    img1, im_color, im_with_contours = image_read(uploaded_image)
    col = st.beta_columns(3)
    col[0].image(img1, caption='Original Image')
    col[1].image(im_color, caption='Transformed Image')
    col[2].image(im_with_contours, caption='Contoured Image')
    model_load(sel2,img1)
