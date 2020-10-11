'''
    python predict_traits_from_speech.py text_dir
    To predict a score on given a document/user's speech
'''
import pickle
import happyfuntokenizing as emo_tokenizer
from nltk.collocations import *
import numpy as np
import pandas as pd
import sys

tok = emo_tokenizer.Tokenizer(preserve_case=False)
file_dir = sys.argv[1]

with open(file_dir, 'r') as file:
    text_file = file.read().replace('\n', '')

uni_id = pickle.load(open('model_src/uni_id', 'rb'))
bi_id = pickle.load(open('model_src/bi_id', 'rb'))
tri_id = pickle.load(open('model_src/tri_id', 'rb'))

#general usr dict first
usr_dict = []
uni_dict = {}
bi_dict = {}
tri_dict = {}
uni_wc = 0
bi_wc = 0
tri_wc = 0
for text in text_file:
    tokens = tok.tokenize(text)
    uni_wc += len(tokens)
    bi_wc += len(tokens)-1
    tri_wc += len(tokens)-2
    for i in range(0, len(tokens)):
        if tokens[i] in uni_id:
            if tokens[i] in uni_dict:
                uni_dict[tokens[i]] += 1
            else:
                uni_dict[tokens[i]] = 1
        if i < len(tokens) -1:
            bigram = " ".join([tokens[i], tokens[i+1]])
            if bigram in bi_id:
                if bigram in bi_dict:
                    bi_dict[bigram] += 1
                else:
                    bi_dict[bigram] = 1
        if i < len(tokens) -2:
            trigram = " ".join([tokens[i], tokens[i+1], tokens[i+2]])
            if trigram in tri_id:
                if trigram in tri_dict:
                    tri_dict[trigram] += 1
                else:
                    tri_dict[trigram] = 1

#usr normalized & binary word count dicts

usr_nwc_dict = []
usr_bwc_dict = []
    
arr_uni_n = np.zeros(len(uni_id))
arr_bi_n = np.zeros(len(bi_id))
arr_tri_n = np.zeros(len(tri_id))
arr_uni_b = np.zeros(len(uni_id))
arr_bi_b = np.zeros(len(bi_id))
arr_tri_b = np.zeros(len(tri_id))
for word in uni_dict:
    arr_uni_n[uni_id[word]] = uni_dict[word]/uni_wc
    arr_uni_b[uni_id[word]] = 1
for word in bi_dict:
    arr_bi_n[bi_id[word]] = bi_dict[word]/bi_wc
    arr_bi_b[bi_id[word]] = 1
for word in tri_dict:
    arr_tri_n[tri_id[word]] = tri_dict[word]/tri_wc
    arr_tri_b[tri_id[word]] = 1 
arr_n = np.concatenate((arr_uni_n, arr_bi_n, arr_tri_n))
usr_nwc_dict = arr_n
arr_b = np.concatenate((arr_uni_b, arr_bi_b, arr_tri_b))
usr_bwc_dict = arr_b

lda_components = pickle.load(open('model_src/lda_components_arr','rb'))
sparse_feature_names = pickle.load(open('model_src/sparse_feature_names','rb'))
col_sum = []
for j in range(0, lda_components.shape[1]):
    col_sum.append(sum(lda_components[:,j]))

#Generate p(usr, topic) scores
pLDA = {}
score_arr = []
wdict = uni_dict
for i in range(0, lda_components.shape[0]):
    topic_arr = lda_components[i,:]
    idx = np.argpartition(topic_arr, -1000) #Check most relevant 1000 words in each topic
    score = 0
    for windex in idx[-1000:]:
        if sparse_feature_names[windex] in wdict:
            score += wdict[sparse_feature_names[windex]]*lda_components[i, windex]/col_sum[windex]
    score_arr.append(score)

bword_matrix = usr_bwc_dict.reshape(1, -1)
nword_matrix = usr_nwc_dict.reshape(1, -1)
topic_matrix = np.array(score_arr).reshape(1, -1)

traits = ['ope', 'con','ext','agr', 'neu']
scores = []

for trait_i in range(0, len(traits)):
    #univariate selection & pca on bword and nword matrices
    univarsel_b = pickle.load(open('model_src/bword_univarsel_'+traits[trait_i],'rb'))
    pca_b = pickle.load(open('model_src/bword_pca_'+traits[trait_i],'rb'))
    univarsel_n = pickle.load(open('model_src/nword_univarsel_'+traits[trait_i],'rb'))
    pca_n = pickle.load(open('model_src/nword_pca_'+traits[trait_i],'rb'))
    Xb_new = univarsel_b.transform(bword_matrix)
    Xb_new1 = pca_b.transform(Xb_new)
    Xn_new = univarsel_n.transform(nword_matrix)
    Xn_new1 = pca_n.transform(Xn_new)
    
    #concatenation sequence: topic_matrix[i,:], nword_[i,:], bword[i,:]
    input_array = np.concatenate((topic_matrix[0,:], Xn_new1[0,:], Xb_new1[0,:]))
    
    ridge = pickle.load(open('model_src/ridge_'+traits[trait_i],'rb'))
    score = ridge.predict(np.array(input_array).reshape(1, -1))[0]
    scores.append(score)
    
    print(traits[trait_i] + " :" + '{:.2f}'.format(scores[trait_i]) )
