import streamlit as st
import pandas as pd
import numpy as np
from IPython.display import Image
from matplotlib import pyplot as plt
pd.options.display.float_format = '{:,.2f}'.format
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os, sys
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import torch
from PIL import Image
from nltk.corpus import stopwords 
from nltk import download 
# from pyemd import emd
# import gensim
# from gensim.models import Word2Vec
import streamlit.components.v1 as components
import base64

download('stopwords')
stop_words = stopwords.words('english')

from google.cloud import vision
import io

import warnings
warnings.simplefilter("ignore")

import os, cv2
# HtmlFile = open(r"C:\AIP11\iapp · Streamlit.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# print(source_code)
# components.html(source_code, height= 1600, width=1600)
def loadusingcv(img):
    im = Image.open(img)
    # img =  cv.imread(im)
    return cv2.cvtColor(np.array(im), cv2.COLOR_BAYER_BG2RGB)
st.title('AUTOMATIC ANSWER CHECKER')
ims1=[]
ims2=[]
n11= st.number_input('Enter number of images ' , key = "n11")
n1 = int(n11)
for i in range(0, n1):
    im1 = st.file_uploader("Teacher1", type=['JPG' , 'JPEG' ,'PNG'],accept_multiple_files=False, key=i)

    if im1 is not None:
        my_img = Image.open(im1)
        im1 = np.array(my_img)
    ims1.append(im1)
# im1 = loadusingcv(image1)
n22= st.number_input('Enter number of images ' , key = "n22")
n2 = int(n22)
for j in range(0, n2):
    im2 = st.file_uploader("Student1" , type =['JPG' , 'JPEG' ,'PNG'] ,accept_multiple_files=False, key=n1+j)
    # im2 = loadusingcv(image2)

    if im2 is not None:
        my_img2 = Image.open(im2)
        im2 = np.array(my_img2)
    ims2.append(im2)
from sentence_transformers import SentenceTransformer , util


embed =  SentenceTransformer('stsb-mpnet-base-v2')
# with open('D:\AIP11\model.pkl', 'rb') as handle:
#     model = pickle.load(handle)
# os.chdir(r"D:\AIP11\testfiles")

         
# fileList = [x for x in os.listdir() if 'png'  in x.lower()  ]/

def segmentation(im):
    
    #img = cv2.imread(im) 
    
    #convert image to greyscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    
    # set threshold to remove background noise
    thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    
    # define rectangle structure (line) to look for: width 100, hight 1. This is a 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200,1))
    
    # Find horizontal lines
    lineLocations = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    

    #plt.figure(figsize=(24,24))
    #plt.imshow(lineLocations, cmap='Greys')
    
    df_lineLocations = pd.DataFrame(lineLocations.sum(axis=1)).reset_index()
    df_lineLocations.columns = ['rowLoc', 'LineLength']
    #df_lineLocations[df_lineLocations['LineLength'] > 0]
    
    df_lineLocations['line'] = 0
    df_lineLocations['line'][df_lineLocations['LineLength'] > 100] = 1

    df_lineLocations['cumSum'] = df_lineLocations['line'].cumsum()
    #df_lineLocations.head()
    
    import pandasql as ps

    query = '''
    select row_number() over (order by cumSum) as SegmentOrder
    , min(rowLoc) as SegmentStart
    , max(rowLoc) - min(rowLoc) as Height
    from df_lineLocations
    where line = 0
    --and CumSum !=0
    group by cumSum
    '''

    df_SegmentLocations  = ps.sqldf(query, locals())
    #df_SegmentLocations
    
    
    im22 = im.copy()
    segments = []

    for i in range(len(df_SegmentLocations)):
        y = df_SegmentLocations['SegmentStart'][i]
        h = df_SegmentLocations['Height'][i]

        cropped = im22[y:y + h, 0:lineLocations.shape[1]] 
        segments.append(cropped)
        # plt.figure(figsize=(8,8))
        # plt.imshow(cropped)
        # plt.title(str(i+1))        

    return segments

    

    
    
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\AIP11\Tanya-369817-16e0e24e5552.json"
# from google.cloud.vision import types

def CloudVisionTextExtractor(handwritings):
    # convert image from numpy to bytes for submittion to Google Cloud Vision
    _, encoded_image = cv2.imencode('.png', handwritings)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    
    # feed handwriting image segment to the Google Cloud Vision API
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=image)
    
    return response

def getTextFromVisionResponse(response):
    texts = []
    for page in response.full_text_annotation.pages:
        for i, block in enumerate(page.blocks):  
            for paragraph in block.paragraphs:       
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    texts.append(word_text)

    return ' '.join(texts)    


# img1 = 
# segments1 = segmentation(img1)
# for i in segments1:
#     segment = i
#     text = CloudVisionTextExtractor(segment)
#     tt = getTextFromVisionResponse(text)
#     #  type(text)
#     # print(tt)
#     # print("\n")
#     f = open(file1.txt, 'w')
#     f = f.write(tt)
#     f.close()

# img2 = 
# segments2 = segmentation(img2)
# for i in segments2:
#     segment = i
#     text = CloudVisionTextExtractor(segment)
#     tt = getTextFromVisionResponse(text)
#     #  type(text)
#     # print(tt)
#     # print("\n")
#     g = open(file2.txt, 'w')
#     g = f.write(tt)
#     g.close()


def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity
negative = ["not" , "without","against","bad","useless","no","dislike","hate"]

def semantic_similarity(actual_answer , given_answer) :
    actual = actual_answer.lower().split(".")
    given = given_answer.lower().split(".")
    
    
    
    sim_checker = actual 
    
    not_matching_semantics = list()
    
    semantic_1 = 0   # Actual_answer
    semantic_2 = 0   # Given_answee
    
    actual_embed_list = list()
    given_embed_list = list()
    
    
    
    for z in range(len(actual)) :
        list_actual = list()  
        list_actual.append(actual[z])
        actual_embed_list.append(embed(list_actual))
        #print(actual_embed_list[z].shape)
    
    for z in range(len(given)) :
        
        semantic_1 = 0
        semantic_2 = 0 
        list_given = list()
        list_given.append(given[z])
        embed_z = embed(list_given)
        
        
        sim_check = sim_checker.copy() 
        sim_check.append(given[z]) 
        
        sen_em = embed(sim_check)
        
        similarity_matrix = cos_sim(np.array(sen_em))
        
        similarity_matrix_df = pd.DataFrame(similarity_matrix) 
        
        cos_list = list(similarity_matrix_df[len(similarity_matrix_df) - 1]) 
        cos_list = cos_list[:len(cos_list)-1]
        #print(cos_list)
        
        index = cos_list.index(max(cos_list))
        
        actual_check = actual[index]
        actual_check = actual_check.split()
        for i in range(len(actual_check) - 1) :
            if(actual_check[i] in negative and actual_check[i+1] in negative) :
                semantic_1 += 1 
            elif(actual_check[i] in negative and actual_check[i+1] not in negative) :
                semantic_1 -= 1 
                
        
        


        answer_given = given[z].split()
        for i in range(len(answer_given) - 1) :
            if(answer_given[i] in negative and answer_given[i+1] in negative) :
                semantic_2 += 1 
            elif(answer_given[i] in negative and answer_given[i+1] not in negative) :
                semantic_2 -= 1 

                
        
        if(semantic_1 == 0 and semantic_2 == 0) :
            pass
            # """
            # Well and good
            # """
        elif(semantic_1 < 0  and semantic_2 >= 0) :
            not_matching_semantics.append(list([actual[index],given[z]]))
            embed_z*=(-1)
            
        
        elif(semantic_1 >= 0 and semantic_2 < 0 ) :
            not_matching_semantics.append(list([actual[index],given[z]]))
            embed_z*=(-1)
            
        
        
        #print(semantic_1,semantic_2,actual[index],given[z])
        
        
        given_embed_list.append(embed_z)
        
        
        
    
    #print(np.array(actual_embed_list).shape)
    actual_embed = actual_embed_list[0] 
    #print(actual_embed.shape) 
    
    for i in range(len(actual_embed_list)-1) :
        #print(actual_embed_list[i+1].shape)
        actual_embed += actual_embed_list[i+1]
        
    given_embed = given_embed_list[0] 
    for i in range(len(given_embed_list) - 1) :
        given_embed += given_embed_list[i+1] 
        
        
    
            
    actual_embed = np.array(actual_embed).reshape(512)
    given_embed = np.array(given_embed).reshape(512) 
    sem_checker = list([actual_embed,given_embed]) 
    answer = pd.DataFrame(cos_sim(sem_checker))
            
        
    return not_matching_semantics , answer[0][1]

def WMD(actual_answer , given_answer,model) :
    
    actual_answer = actual_answer.lower().split()
    actual_answer = [w for w in actual_answer if w not in stop_words]
    
    
    given_answer = given_answer.lower().split()
    given_answer = [w for w in given_answer if w not in stop_words]
    
    
    return model.wmdistance(given_answer,actual_answer)
    
def score(given_answer , actual_answer,model) :
    
    actual_answer_emb = model.encode(actual_answer , convert_to_tensor=True)
    given_answer_emb = model.encode(given_answer , convert_to_tensor=True)

# Cosine Similarity
    

    similarity = util.cos_sim(actual_answer_emb, given_answer_emb)
    # distance = WMD(actual_answer2,given_answer2,model)
    

    
    if(similarity > 0) :
        # if(distance == 0) :
        #     return 1 
        return similarity
    else :
        return 0
inputmarks=[]


finalmarks =[]
segments1 = []
segments2 =[]
for i in range(0, n1):
    seg1 = segmentation(ims1[i])
    segments1 = segments1 + seg1
for i in range(0, n2):
    seg2 = segmentation(ims2[i])
    segments2 = segments2 + seg2

# segments1 = segmentation(im1)
# segments2 = segmentation(im2)
      


for i in range(0, len(segments1)):
    x = st.number_input('marks for question' , key = n1+n2+i)
    inputmarks.append(x)

for i in range(0,len(segments1)):
    segment1 = segments1[i]
    segment2 = segments2[i] 
    text1 = CloudVisionTextExtractor(segment1)
    tt1 = getTextFromVisionResponse(text1)
    text2 = CloudVisionTextExtractor(segment2)
    tt2 = getTextFromVisionResponse(text2)
    x = score(tt1,tt2,embed)
    finalmarks.append(x)

if st.button('Analyze'):
    for i in range (0, len(finalmarks)):
        st.write("marks for answer" , i+1  , "are " ,round(float(finalmarks[i])*inputmarks[i],2))
    print(finalmarks)