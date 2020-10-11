'''
    This code produces features used for training the Ridge model
'''
from collections import defaultdict
import pickle
import pandas as pd 
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

'''
...data/
usr_to_msg: dictionary of documents/posts by user
usr_norm_words: dictionary with normalized word counts by user
usr_bin_words: dictionary with binary representation of words by user
usr_list: list used to map from dictionary to array
usr_topic_scores_200: dictionary with [score_topic_1, ...] per user
usr_to_agr, etc: dict[usrid] = score
val_id: list of usrids in the validation/test set
val_usr_list: list used to map from dicationry to array
'''

'''
# Generate y values
f = pd.read_csv('personality.csv')
userids = f['userid'].values
neu = f['neu'].values
print(userids.shape)
print(neu.shape)

usr_to_neu = defaultdict(list)

for i in range(userids.shape[0]):
    usr_to_neu[userids[i]] = neu[i]

print(len(usr_to_neu))
with open('usr_to_neu', 'wb') as f2:
    pickle.dump(usr_to_neu, f2)
'''

# Dimentionality reduction
usr_norm_words = pickle.load(open('data/usr_norm_words','rb'))
usr_bin_words = pickle.load(open('data/usr_bin_words','rb'))
usrs = pickle.load(open('data/usr_list','rb'))
usr_to_neu = pickle.load(open('data/usr_to_neu','rb'))
neu_matrix = np.array([usr_to_neu[i] for i in usrs])

bword_matrix = np.array([usr_bin_words[i] for i in usrs])

univarsel_b = SelectKBest(f_regression, k=5000)
univarsel_b.fit(bword_matrix, neu_matrix) 
Xb_new = univarsel_b.transform(bword_matrix)

with open("fitted_model/bword_univarsel_neu","wb") as fp:
    pickle.dump(univarsel_b,fp)
    
pca_b = PCA(n_components=2453)
pca_b.fit(Xb_new)
Xb_new1 = pca_b.transform(Xb_new)

with open("fitted_model/bword_pca_neu","wb") as fp:
    pickle.dump(pca_b,fp)

with open("fitted_model/bword_neu_mat_2453","wb") as fp:
    pickle.dump(Xb_new1,fp)


nword_matrix = np.array([usr_norm_words[i] for i in usrs])
univarsel_n = SelectKBest(f_regression, k=5000)
univarsel_n.fit(nword_matrix, neu_matrix)
Xn_new = univarsel_n.transform(nword_matrix)

with open("fitted_model/nword_univarsel_neu","wb") as fp:
    pickle.dump(univarsel_n,fp)

pca_n = PCA(n_components=2453)
pca_n.fit(Xn_new)
Xn_new1 = pca_n.transform(Xn_new)

with open("fitted_model/nword_pca_neu","wb") as fp:
    pickle.dump(pca_n,fp)

with open("fitted_model/nword_neu_mat_2453","wb") as fp:
    pickle.dump(Xn_new1,fp)

# Load input matrices
usr_topic_scores = pickle.load(open('data/usr_topic_scores_200','rb'))
topic_matrix = np.array([usr_topic_scores[i] for i in usrs])

#nword_con_mat_2453 = pickle.load(open('nword_con_mat_2453','rb'))
#bword_con_mat_2453 = pickle.load(open('bword_con_mat_2453','rb'))

input_arr = []
for i in range(0, 49921):
    #arr = np.concatenate((topic_matrix[i,:], nword_con_mat_2453[i,:], bword_con_mat_2453[i,:]))
    arr = np.concatenate((topic_matrix[i,:], Xn_new1[i,:], Xb_new1[i,:]))
    input_arr.append(arr)

input_arr = np.array(input_arr)
clf = Ridge(alpha=1.0)
clf.fit(input_arr, neu_matrix) 
print(clf.score(input_arr, neu_matrix))

with open("fitted_model/ridge_neu","wb") as fp:
    pickle.dump(clf,fp)

'''
# Load input directly to fit new models & tune parameters

clf = pickle.load(open('ridge_ext','rb'))
usr_to_ext = pickle.load(open('usr_to_ext','rb'))
ext_matrix = np.array([usr_to_ext[i] for i in usrs])

usr_norm_words = pickle.load(open('usr_norm_words','rb'))
usr_bin_words = pickle.load(open('usr_bin_words','rb'))
usrs = pickle.load(open('usr_list','rb'))
usr_topic_scores = pickle.load(open('usr_topic_scores_200','rb'))

#generate entries for testing
bword_matrix = np.array([usr_bin_words[i] for i in usrs])
nword_matrix = np.array([usr_norm_words[i] for i in usrs])
topic_matrix = np.array([usr_topic_scores[i] for i in usrs])

#univariate selection & pca on bword and nword matrices - for example
univarsel_b = pickle.load(open('bword_univarsel_ext','rb'))
pca_b = pickle.load(open('bword_pca_ext','rb'))
univarsel_n = pickle.load(open('nword_univarsel_ext','rb'))
pca_n = pickle.load(open('nword_pca_ext','rb'))
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

clf.score(input_arr, ext_matrix)

'''
