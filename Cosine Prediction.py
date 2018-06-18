
# coding: utf-8

# In[1]:


import pandas as pd

#read the file
df = pd.read_csv('/home/admin/Downloads/India Census 2011 data.xlsx - train data.csv')


df= df[['Name']]


df.head()


# In[6]:


#read the file
test = pd.read_csv('/home/admin/Downloads/India Census 2011 data.xlsx - test data.csv')

test= test[['lower']]
test.head()


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
a=df['Name'].count()
list_new=[]
for i in test.index :
    my_list=[]
   
    for j in range(0,len(df)) :
         documents= (test['lower'][i],df['Name'][j])
         
         
         tfidf_vectorizer = TfidfVectorizer(analyzer = "char")
         tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
         
         cs = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
         
         my_list.append(cs)
        
    df1=pd.DataFrame.from_records(my_list)
    list=[]
    for k in df1[0] : 
            
            list.append(k[1])
         
    df1=pd.DataFrame(list)
    df_new = pd.concat([df, df1], axis=1)
    df_new.columns = df_new.columns.map(str)
    df_new = df_new.rename(columns={'0': 'Cosine'})
    
    list_new.append(df_new['Name'][df_new['Cosine'] == max(df_new['Cosine'])])
    
        


# In[11]:


_list = tuple(zip(*list_new))[0]
se = pd.Series(_list)
test['Predicted'] = se.values


# In[8]:


test.to_csv('CosinePrediction.csv')


#     
