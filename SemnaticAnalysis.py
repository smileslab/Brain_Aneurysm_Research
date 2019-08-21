
# coding: utf-8

# In[5]:


import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim import models

import pandas as pd
#import Bio
#import docx
import PyPDF2
import os


# In[8]:


list_pdfchapter=os.listdir(path='C:\\Users\\T8828FA\\Knowledgegraph\\IABook\\pdf')


# In[7]:


pwd


# ### 1. Create Text file from chapter pdf File

# In[9]:


f= open("corpus.txt","w+")

for chap in list_pdfchapter:
    pdfFileObj = open('C:\\Users\\T8828FA\\Knowledgegraph\\IABook\\pdf\\'+chap, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    numpages=pdfReader.getNumPages()
    for i in range(1,numpages,1):
        try:
            pageObj = pdfReader.getPage(i)
            txt=pageObj.extractText()
            f.write(txt)
        except Exception as e: 
            print(e)
            print('Error Page number',i)
            continue
f.close() 


# ### 2. Append extra training data from PubMed Abstracts provided by Dr. Afzal

# In[10]:


f_append=open("PubMed Abstracts.txt","r", errors='ignore')
str_f_append=f_append.read()
f_append.close()
with open("corpus.txt", "a") as myfile:
    myfile.write(str_f_append)


# In[11]:


f= open("corpus.txt","r")
contents =f.read()
#print(contents)


# In[12]:


## Read Concept File


# In[13]:


df_concept=pd.read_excel('Final_concept v1.2.xlsx',sheetname='Final_concept_list')
def concatenateConcepts(cols):
    '''
    @Author - Fakhare Alam
    '''
    concept = cols[0]
    concept_list=concept.split()
    return '_'.join(concept_list).lower()
df_concept['Concept_concat'] = df_concept[['Concept']].apply(concatenateConcepts,axis=1)


# ## Read the input file

# In[14]:


input_file='corpus.txt'


# In[15]:


fl= open("corpus_line.txt","w+")
def readInputFile(input_file):
    bigram_token = []
    f=open(input_file,mode='r')
    series_concept=df_concept['Concept']
    list_concept=series_concept.tolist()
    series_concept_con=df_concept['Concept_concat']
    list_concept_con=series_concept_con.tolist()
    #yield preprocess_documents(f)

    for i, line in enumerate (f):
        # Concatenate with _
        line=remove_stopwords(line)
        line=line.lower()
        
        for i in range(0,len(list_concept),1):
            line=line.replace(list_concept[i].lower(),list_concept_con[i].lower())
        fl.write(line)
        #print(line)    
        
        yield gensim.utils.simple_preprocess (line,max_len=50)
        
        
documents = list (readInputFile (input_file))


# In[ ]:


#documents


# ## Train the model

# In[17]:


model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=1, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)


# ## Read concept Excel And Pre-process

# In[18]:


series_concept=df_concept['Concept']
list_concept=series_concept.tolist()

series_concept_con=df_concept['Concept_concat']
list_concept_con=series_concept_con.tolist()


# ## Find Similarity between concept Pairs

# In[19]:


def conceptScore(cols):
    '''
    @Author - Fakhare Alam
    '''
    concept = cols[0]
    try:
        concept_score=model.wv.similarity(w1=concept,w2="aneurysm")
    except:
        concept_score=0.0
    return concept_score


# In[20]:


df_concept['Concept_Score'] = df_concept[['Concept_concat']].apply(conceptScore,axis=1)


# In[21]:


df_concept.to_excel('Concept_Score.xlsx')

