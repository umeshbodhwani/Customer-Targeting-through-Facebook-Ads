'''
    This code tests validation/testing set data on the models
'''
import pickle
import happyfuntokenizing as emo_tokenizer
from nltk.collocations import *
import numpy as np
import pandas as pd

#Generate validation set
usr_to_msg = pickle.load(open('data/usr_to_msg1','rb'))

val_id = pickle.load(open('data/val_id','rb'))
print(len(val_id))

val_set = {}
count = 0
for key, value in val_id.items():
    if key in usr_to_msg:
        val_set[key] = usr_to_msg[key]
    else:
        count += 1
        print(count)

bi_id = pickle.load(open('model_src/bi_id','rb'))
uni_id = pickle.load(open('model_src/uni_id','rb'))
tri_id = pickle.load(open('model_src/tri_id','rb'))

tok = emo_tokenizer.Tokenizer(preserve_case=False)

#general usr dict first
counter = 0
usr_dict = {}
for usr, texts in val_set.items():
    uni_dict = {}
    bi_dict = {}
    tri_dict = {}
    uni_wc = 0
    bi_wc = 0
    tri_wc = 0
    for text in texts:
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
    usr_dict[usr] = [uni_dict, uni_wc, bi_dict, bi_wc, tri_dict, tri_wc]
    
    if counter %1000 == 0:
        print(counter)
    counter +=1

#usr normalized & binary word count dicts
counter = 0
usr_nwc_dict = {}
usr_bwc_dict = {}
for usr, dict_list in usr_dict.items():
    uni_dict, uni_wc, bi_dict, bi_wc, tri_dict, tri_wc = dict_list
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
    usr_nwc_dict[usr] = arr_n
    arr_b = np.concatenate((arr_uni_b, arr_bi_b, arr_tri_b))
    usr_bwc_dict[usr] = arr_b
    
    if counter %1000 == 0:
        print(counter)
    counter +=1
    
    
lda_components = pickle.load(open('model_src/lda_components_arr','rb'))
sparse_feature_names = pickle.load(open('model_src/sparse_feature_names','rb'))

col_sum = []
for j in range(0, lda_components.shape[1]):
    col_sum.append(sum(lda_components[:,j]))

#Generate p(usr, topic) scores
pLDA = {}
counter = 0
for usr in usr_dict:
    score_arr = []
    wdict = usr_dict[usr][0]
    for i in range(0, lda_components.shape[0]):
        topic_arr = lda_components[i,:]
        idx = np.argpartition(topic_arr, -1000) #Check most relevant 1000 words in each topic
        score = 0
        for windex in idx[-1000:]:
            if sparse_feature_names[windex] in wdict:
                score += wdict[sparse_feature_names[windex]]*lda_components[i, windex]/col_sum[windex]
        score_arr.append(score)
    pLDA[usr] = score_arr
    
    if counter%100 == 0:
        print(counter)
    counter += 1

val_usrs = pickle.load(open('data/val_usr_list','rb'))
#generate entries for testing
bword_matrix = np.array([usr_bwc_dict[i] for i in val_usrs])
nword_matrix = np.array([usr_nwc_dict[i] for i in val_usrs])
topic_matrix = np.array([pLDA[i] for i in val_usrs])

#univariate selection & pca on bword and nword matrices - agr for example
univarsel_b = pickle.load(open('model_src/bword_univarsel_agr','rb'))
pca_b = pickle.load(open('model_src/bword_pca_agr','rb'))
univarsel_n = pickle.load(open('model_src/nword_univarsel_agr','rb'))
pca_n = pickle.load(open('model_src/nword_pca_agr','rb'))
Xb_new = univarsel_b.transform(bword_matrix)
Xb_new1 = pca_b.transform(Xb_new)
Xn_new = univarsel_n.transform(nword_matrix)
Xn_new1 = pca_n.transform(Xn_new)

#test
#concatenation sequence: topic_matrix[i,:], nword_[i,:], bword[i,:]
input_arr = []
for i in range(0, Xb_new.shape[0]):
    arr = np.concatenate((topic_matrix[i,:], Xn_new1[i,:], Xb_new1[i,:]))
    input_arr.append(arr)
input_arr = np.array(input_arr)

ridge = pickle.load(open('model_src/ridge_agr','rb'))
usr_to_agr = pickle.load(open('data/usr_to_agr','rb'))
agr_matrix = np.array([usr_to_agr[i] for i in val_usrs])
print(str(ridge.score(input_arr, agr_matrix)))
