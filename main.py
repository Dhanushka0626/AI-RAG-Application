import os
import tempfile

from tempfile import NamedTemporaryFile

import streamlit as st
import chromadb

from query_data import classify_img, get_most_similar_chunks, create_response
from util import DB_PATH

client_db = chromadb.PersistentClient(path=DB_PATH)

#set title
st.title("tour-guid-ai-assistant-app")

#set header
st.header("Please upload an image")

#upload file
file = st.file_uploader("",type=["jpg","jpeg","png"])

if file:
    #display image
    st.image(file, width=500)
    
    
    
    ##############################
    ### compute agent response ###
    ##############################
    suffix = os.path.splitext(file.name)[1]
    
    with NamedTemporaryFile(dir='.', suffix=suffix, delete=False) as f:
        f.write(file.getbuffer())
        image_path = f.name
        
    clf = classify_img(client_db,image_path)
    clf_ = clf.replace('_', ' ').title()
    
    st.write(f"You are currently looking at the **{clf_}** !\n Is there anything you would lime to know about it?")
    
    #text input
    user_qestion = st.text_input(f"Ask a question about **{clf_}**:")
        
    # write agent response
    if user_qestion and user_qestion != "":
        with st.spinner(text="In progress..."):
            
            chunks, metadatas = get_most_similar_chunks(client_db,user_qestion,clf)
            response, source = create_response(chunks,metadatas,user_qestion)
            
            st.write(response)
            
            
            
    
    