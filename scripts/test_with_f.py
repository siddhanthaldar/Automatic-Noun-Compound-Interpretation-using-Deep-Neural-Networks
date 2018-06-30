import tensorflow as tf
import numpy as np
import read
import embedding

def word_idx_matrix(word):

	word_idx = np.zeros((len(word),1))
	count = 0

	for i in word:
		word_idx[count] = word_to_idx[i] - 1
		count += 1

	return word_idx


word_to_idx, idx_to_word,label_to_idx,idx_to_label = read.read_data()
word1, word2 ,train_label = read.read_test_data(idx_to_word, idx_to_label)
embedding_matrix = embedding.CW_embedding(idx_to_word)

num_data = len(word1)
embedding_dim = embedding_matrix.shape[1]
num_label = len(label_to_idx) - 1

#Generate word indices
word1_idx = word_idx_matrix(word1)
word2_idx = word_idx_matrix(word2)

#Convert to int type
word1_idx = word1_idx.astype(int)
word2_idx = word2_idx.astype(int)

#one hot representation of labels
labels_one_hot = np.zeros((len(train_label),num_label))
for i in range(len(train_label)):
	idx = int(label_to_idx[train_label[i]] - 1)
	labels_one_hot[i][idx] = 1

# ********************* Model *************************
#Necessary Functions
def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1, dtype = tf.float64), dtype = tf.float64)

def bias_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1, dtype = tf.float64), dtype = tf.float64)

def mul_mat(X,W,b):
	return tf.matmul(X,W) + b

X1 = tf.placeholder(tf.int64, shape = ([None, 1]))
X2 = tf.placeholder(tf.int64, shape = ([None, 1]))
Y = tf.placeholder(tf.float64, shape = ([None, num_label]))  #labels

# For fine tuning embedded layer
embedding_layer = tf.Variable(tf.constant(embedding_matrix))
embed_vec1 = tf.nn.embedding_lookup(embedding_layer, X1)
embed_vec1 = tf.squeeze(embed_vec1)   # To remove dimensions of size 1

embed_vec2 = tf.nn.embedding_lookup(embedding_layer, X2)
embed_vec2 = tf.squeeze(embed_vec2)   # To remove dimensions of size 1

#hidden layer
W_h1 = weight_variable((embedding_dim, embedding_dim))
b_h1 = bias_variable([1, embedding_dim])
W_h2 = weight_variable((embedding_dim, embedding_dim))
b_h2 = bias_variable([1, embedding_dim])

h = tf.nn.relu(mul_mat(embed_vec1,W_h1, b_h1) + mul_mat(embed_vec2,W_h2, b_h2))

# output layer
W_out = weight_variable([embedding_dim, num_label])
b_out = bias_variable([1, num_label])

output = mul_mat(h,W_out,b_out)   
	
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = output))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted = tf.argmax(output,1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
		#***************Restore weights ***************
	sess.run(init)
	saver= tf.train.Saver([embedding_layer,W_h1,b_h1,W_h2,b_h2, W_out, b_out])
	saver.restore(sess, "/home/sid/virtual_env/nlp/proj/Compound-Noun-Interpretation/wgts_CW_f.cpkt")
	print("Restored")

	#*************** Predict *********************
	with tf.device('/gpu:0'):
		print(sess.run(cost, feed_dict={X1: word1_idx, X2: word2_idx, Y: labels_one_hot}))
		print('prediced =', predicted.eval(feed_dict={X1: word1_idx, X2: word2_idx, Y: labels_one_hot}))
		print('original = ', tf.argmax(Y,1).eval(feed_dict={X1: word1_idx, X2: word2_idx, Y: labels_one_hot}))
		print('accuracy = ', sess.run(accuracy, feed_dict={X1: word1_idx, X2: word2_idx, Y: labels_one_hot}))

	sess.close()		

