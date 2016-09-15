# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:56:17 2016

@author: ajit
"""
#%%
from pymongo import MongoClient
import gensim,numpy as np,time
from scipy.spatial import distance
import nltk
import pandas as pd


"""Data Cleaning"""                                                                                                                                                                                                                                                                                                                                                                                                             

import re
import itertools as it 





flag=[0]
flag[0]=0
""" Function defination for getting representation data"""
def trv(data,rep):
    if flag[0]==0:
        data=[]
    data.append(rep['name'])
    if rep.get("children") == None:
        flag[0]=1
        return data
    else:
        child=len(rep['children'])
        for i in range(0,child):
            flag[0]=1
            trv(data,rep['children'][i])
        return data

"""
Data Extraction from Input Jsons
"""
""" Functtion to clean the text """
def clean_text(text_list):
    text_list=" ".join(text_list)    
    text_list=re.sub(r'<[^>]*>',"", text_list)      #doubt
    text_list=re.sub(r'&nbsp;', ' ',text_list) 
    text_list=re.sub(r'[^A-Za-z]+',' ',text_list)
    text_list=re.sub(r'\d',"", text_list)
    return text_list

data = []
keyq=[]
gaps=[]
project=[]
rep=[]
rep_data=[]
dummy=[]

        
 
client = MongoClient()

db=client.muuniverse
db.authenticate('ajit','welcome123')


hm=list(db.hypotheses.find())
project=list(db.projects.find())
data=list(db.situations.find())
keyq=list(db.keyquestions.find())
gaps=list(db.gaps.find())
rep=list(db.representations.find())

       


out={}
out2=[]
out1={}
invalid=[]
inactive=0
empkey=[]

db.train_data.drop()
collection=db.train_data

for j in range(0,len(data)):    # Select number of lines of data 
    key=data[j]["project_id"]

    if data[j].get("situation") == None or data[j].get("future_state")== None or gaps[j].get("gap")== None or keyq[j].get("keyquestion")== None:
        invalid.append(key)
        continue
    if project[j]['active'] == 0:
        inactive+=1
        continue
    
    d=[data[j]['situation'], data[j]['future_state'],gaps[j]["gap"], keyq[j]["keyquestion"] ]
    design_data=clean_text(d)
    
    
    if len(design_data) == 1:
        empkey.append(key)
        continue
    
    """ representation"""
    flag[0]=0
    rep_data=trv(dummy,rep[j]['representation'][0])    
    rep_data=clean_text(rep_data)
    
    """ HM """
    
    hm_col=[[ [hm[j]["hypothesis_details"][n]['end_question']] , [hm[j]["hypothesis_details"][n]['factor_bucket'][k]['factor'] for k in range(0,len(hm[j]["hypothesis_details"][n]['factor_bucket']))], [hm[j]["hypothesis_details"][n]['hypothesis_tests'][m]['test'] for m in range(0,len(hm[j]["hypothesis_details"][n]['hypothesis_tests']))]] for n in range(0,len(hm[j]["hypothesis_details"])) ]    
        
    temp=list(it.chain.from_iterable(hm_col))
    hm_data=list(it.chain.from_iterable(temp))
    hm_data=clean_text(hm_data)
    
    all_data = design_data + rep_data + hm_data + ' '
    out1[key]={all_data}
    out2.append(str(all_data))
    db.train_data.insert({"Trained_data":all_data})

with open('/home/ajit/Downloads/t4', 'w') as f:
    f.writelines('{}\t{}\n'.format(k,v) for k, v in out1.items())

with open('/home/ajit/Downloads/t2','w') as f1:
    f1.writelines('%s ' % item for item in out2)

#return (f1,f2)
    
print "Finished"


#%%





#%%

"""Get Vector Representations"""
def get_vectors(file_name,C,model,vec_size):    
    """Getting the list of stop_words"""
    SW = set()
    for line in open('/home/ajit/Applications/WMD Integration/stop_words.txt'):
        line = line.strip()
        if line != '':
            SW.add(line)

    stop = list(SW)
    
    
    StopWords = nltk.corpus.stopwords.words("english")
    extrasw=['%','.',',','$',"'s", '(',')',':','[',']',"set([u'","'])"]    #List extra stopwords
    StopWords.extend(extrasw)
    stop.extend(StopWords)


    f = open(file_name)
    if len(C) == 0:
        C = np.array([], dtype=np.object)
    num_lines = sum(1 for line in open(file_name))

    y = np.zeros((num_lines,))

    X = np.zeros((num_lines,), dtype=np.object)

    BOW_X = np.zeros((num_lines,), dtype=np.object)
    count = 0
    

    the_words = np.zeros((num_lines,), dtype=np.object)
    
    """Split the file into words"""
    for line in f:
        
        T = line.split('\t')
        classID = T[0]
        C = np.append(C,classID)
        y[count] = len(C)
        W = line.split()
        
        F = np.zeros((vec_size,len(W)-1))
        inner = 0
        
        word_order = np.zeros((len(W)-1), dtype=np.object)
        bow_x = np.zeros((len(W)-1,))
        
        """Obtain the vector representation of words and remove the stopwords"""
        for word in W[1:len(W)]:
            
            if word in stop:
                word_order[inner]=''
                continue
            if word in word_order:
                IXW = np.where(word_order==word)
                bow_x[IXW] += 1
                word_order[inner] = ''
            else:
                word_order[inner] = word
                bow_x[inner] += 1
                F[:,inner] = model[word]
                
            inner = inner + 1

        Fs = F.T[~np.all(F.T == 0,axis=1)]          #extracting non-zero values
        word_orders = word_order[word_order != '']  #words
        word_orders = word_orders[word_orders !=0]
        bow_xs = bow_x[bow_x != 0]                  #frequency of the word
        X[count] = Fs.T                             #count is the sentence number
        the_words[count] = word_orders
        BOW_X[count] = bow_xs
        count = count + 1;
    return (X,BOW_X,y,C,the_words)




localtime = time.asctime( time.localtime(time.time()) )
print "Before :", localtime
"""Train the dataset"""
model = gensim.models.Word2Vec(min_count=1,size=300,window = 5,workers=4,sg=1)
sentences = gensim.models.word2vec.LineSentence('/home/ajit/Downloads/data')

model.build_vocab(sentences)
model.train(sentences)
localtime = time.asctime( time.localtime(time.time()) )
print "After :", localtime
vec_size = 300

train_dataset = '/home/ajit/Downloads/t3'

"""reading document dataset"""
(X,BOW_X,y,C,words)  = get_vectors(train_dataset,[],model,vec_size)

#%%















#%%
n = np.shape(X)
n = n[0]

for i in range(n):
    bow_i = BOW_X[i]
    bow_i = bow_i / np.sum(bow_i)
    bow_i = bow_i.tolist()
    BOW_X[i] = bow_i
    X_i = X[i].T
    X_i = X_i.tolist()
    X[i] = X_i


    

l=0
m=0
d3=[]
d2=[]

"""Calculating the minimum distance of one document from other document"""
a=pd.DataFrame(columns=('col1','col2','word1','word2'))
def euclid_distance(X,BOW_X,a):
    w=0
    
    for l in range(len(X)):
        for n in range(l+1,len(X)):
            d1=[]
            for m in range(len(X[l])):
                min1=99999
                p=0
                while(p<len(X[n])):
                    d=distance.euclidean(X[l][m],X[n][p])
                    d=d*np.min([BOW_X[l][m],BOW_X[n][p]])
                    if(d<min1):
                        min1=d
                        q=p
                    p+=1
                d1.append(min1)
                if(min1!=0):
                    a.loc[w]=[C[l],C[n],words[l][m],words[n][q]]
                    w+=1
            d2.append(d1)
            
    for i in range(len(d2)):
        d3.append(sum(d2[i]))
    
    return (d3,a)
            
(d,a)=euclid_distance(X,BOW_X,a)            

#%%

C=C.tolist()
       
"""Taking Cumulative Sum"""

a.to_csv("a.csv")

#a=a.groupby(['col1','col2'])
a['word']=a.word1.str.cat(a.word2,sep='-')
a['col']=a.col1.str.cat(a.col2,sep='-')

#b=pd.DataFrame(columns=('col1','col2','word'))
#%%
   
c=a.groupby(['col1','col2'])['word'].apply(lambda x: ','.join(x)).reset_index()       
c.to_csv("c.csv")

#%%
db.wmds.drop()
collection=db.wmds


#db.wmds.insert({})
"""Just for displaying results"""   
k=0    
for i in range(len(C)):
    for j in range(i+1,len(C)):
        print C[j], ":" , C[i], ":",    d[k]
        
        #b.loc[k]=[C[i],C[j],list(set(words[i]).intersection(words[j]))]
        db.wmds.insert({"targetNode":C[j],"sourceNode":C[i],"wmd":d[k],"common_words":list(set(words[i]).intersection(words[j]))})#print C[i],",",C[j],":",d3[k]
        k+=1

#b.to_csv("b.csv")


