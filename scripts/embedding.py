import os
import numpy as np
import read

def glove_embedding(words):
	GLOVE_DIR = '/home/sid/virtual_env/nlp/proj/embeddings/Glove/'

	embeddings_index = {}

	embedding_dim = 50
	f = open(os.path.join(GLOVE_DIR, 'glove.6B/glove.6B.' + str(embedding_dim) + 'd.txt'))
	# f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'))

	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype = 'float32')
		embeddings_index[word] = coefs

	f.close()

	# print('Found %s word vectors.' % len(embeddings_index))
	# print(len(words))
	embedding_matrix = np.zeros((len(words) + 1, embedding_dim))
	for i,word in words.items():
		# print i
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i-1] = embedding_vector   # (i-1) since count starts from 1 in read.py

	# print(embedding_matrix[5241])

	return embedding_matrix


if __name__ == "__main__":
	_, idx_to_word,_,_ = read.read_data()
	embedding_matrix = glove_embedding(idx_to_word)	
	print embedding_matrix[3145]

