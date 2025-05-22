import streamlit as st 

st.markdown("<h1 style='text-align: center; color:Blue;'>Model Analysis </h1>", unsafe_allow_html=True)

    
    
st.header("Model Details ")
st.markdown("- Model has Accuracy of 91%")
st.markdown("- Model is trained with a simple shallow convolutional network")
st.download_button(
    label="Download Model",
    data="./smile.hdf5",
    file_name="smile_model.hdf5",
)

image1 =  "./media/smile_conv.png"
image2  = "./media/model_plotting.png"


col1,col2  = st.columns(2)


with col1:
    st.header("Model Architecture")
    st.image(image1)

with col2:
    st.header("Model Progress")
    st.image(image2)
    

    
    

