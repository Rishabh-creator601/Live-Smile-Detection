import streamlit as st 


st.title("Model Analysis ")


    
    
st.header("Model Details ")
st.markdown("- Model has Accuracy of 91%")
st.markdown("- Model is trained with a simple shallow convolutional network")
st.download_button(
    label="Download Model",
    data="./smile.hdf5",
    file_name="smile_model.hdf5",
    icon=":material/download:",
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
    

    
    

