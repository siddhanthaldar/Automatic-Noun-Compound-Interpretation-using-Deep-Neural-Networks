import os
import numpy as np
import read
import gensim
from sklearn.decomposition import PCA

def glove_embedding(words):
	GLOVE_DIR = '/home/sid/virtual_env/nlp/proj/embeddings/Glove/'

	embeddings_index = {}

	embedding_dim = 300
	f = open(os.path.join(GLOVE_DIR, 'glove.6B/glove.6B.' + str(embedding_dim) + 'd.txt'))

	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype = 'float32')
		embeddings_index[word] = coefs

	f.close()

	embedding_matrix = np.zeros((len(words), embedding_dim))
	for i,word in words.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i-1] = embedding_vector   # (i-1) since count starts from 1 in read.py

	return embedding_matrix

def CW_embedding(idx_to_word):
	WORD_DIR = "/home/sid/virtual_env/nlp/proj/embeddings/CW/senna/hash/words.lst"
	VECTOR_DIR = "/home/sid/virtual_env/nlp/proj/embeddings/CW/senna/embeddings/embeddings.txt"

	embeddings_index = {}
	embedding_dim = 50

	cw_words = open(WORD_DIR)
	cw_vectors = open(VECTOR_DIR)

	words = cw_words.read().split('\n')

	count = 1
	for line in cw_vectors:
		vals = line.split()
		embeddings_index[words[count-1]] = np.asarray(vals[0:50], dtype = 'float32')
		count += 1

	cw_words.close()

	embedding_matrix = np.zeros((len(idx_to_word), embedding_dim))
	for i,word in idx_to_word.items() : 
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i-1] = embedding_vector

	return embedding_matrix		

def word2vec_embedding(idx_to_word):
	DIR = "home/other/16EE30025/proj/embeddings/GoogleNews-vectors-negative300.bin"

	model = gensim.models.KeyedVectors.load_word2vec_format('./' + DIR, binary=True) 

	embeddings_index = {}
	embedding_dim = 300

	embedding_matrix = np.zeros((len(idx_to_word), embedding_dim))

	for i.word in idx_to_word.items():
		embedding_vector = model.get_vector(word)
		if embedding_vector is not None:
			embedding_matrix[i-1] = embedding_vector

	return embedding_matrix		

def hpca_200_embedding(idx_to_word):
	DIR = "/home/sid/virtual_env/nlp/proj/embeddings/hpca_200/"

	embeddings_index = {}
	embedding_dim = 200

	hpca_words = open(os.path.join(DIR,"vocab.txt"))
	hpca_vectors = open(os.path.join(DIR,"words.txt"))

	words = hpca_words.read().split('\n')

	count = 0

	for line in hpca_vectors:
		vals = line.split()
		embeddings_index[words[count]] = np.asarray(vals[0:embedding_dim], dtype = "float32")
		count += 1

	hpca_words.close()
	hpca_vectors.close()

	embedding_matrix = np.zeros((len(idx_to_word), embedding_dim))
	for i,word in idx_to_word.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			count += 1
			embedding_matrix[i-1] = embedding_vector	

	return embedding_matrix	

def cw_glove_hpca_embedding(idx_to_word):
	embedding_matrix_cw = CW_embedding(idx_to_word)
	embedding_matrix_glove = glove_embedding(idx_to_word)
	embedding_matrix_hpca = hpca_200_embedding(idx_to_word)

	embedding_matrix = np.concatenate((embedding_matrix_cw, embedding_matrix_glove, embedding_matrix_hpca), axis = 1)

	return embedding_matrix



if __name__ == "__main__":
	_, idx_to_word,_,_ = read.read_data()
	embedding_matrix = hpca_200_embedding(idx_to_word)
	print(embedding_matrix.shape)
	print(embedding_matrix[0])