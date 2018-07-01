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
word1, word2 ,label = read.read_cross_val_data(idx_to_word, idx_to_label)
embedding_matrix = embedding.glove_embedding(idx_to_word)

# num_data = len(word1)
embedding_dim = embedding_matrix.shape[1]
num_label = len(label_to_idx) - 1

# #Generate word indices
# word1_idx = word_idx_matrix(word1)
# word2_idx = word_idx_matrix(word2)

# #Convert to int type
# word1_idx = word1_idx.astype(int)
# word2_idx = word2_idx.astype(int)

# #one hot representation of labels
# labels_one_hot = np.zeros((len(train_label),num_label))
# # print(num_label)	
# for i in range(len(train_label)):
# 	idx = int(label_to_idx[train_label[i]] - 1)
# 	labels_one_hot[i][idx] = 1

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
embedding_layer = tf.Variable(tf.constant(embedding_matrix))#, trainable = True)
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
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# Hyperparameters
epochs = 30
num_folds = 10
recording_interval = 500
batch_size = 1000

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(epochs):
		for j in range(num_folds):
			word1_val = word1[j]
			word2_val = word2[j]
			label_val = label[j]

			word1_train = []
			word2_train = []
			label_train = []

			for x in range(num_folds):
				if x != j:
					word1_train += word1[x]
					word2_train += word2[x]
					label_train += label[x]

		num_data = len(word1_train)

		# For Train data
		# Generate word indices
		word1_train_idx = word_idx_matrix(word1_train)
		word2_train_idx = word_idx_matrix(word2_train)

		#Convert to int type
		word1_train_idx = word1_train_idx.astype(int)
		word2_train_idx = word2_train_idx.astype(int)

		#one hot representation of labels
		labels_one_hot_train = np.zeros((len(label_train),num_label))
		# print(num_label)	
		for i in range(len(label_train)):
			idx = int(label_to_idx[label_train[i]] - 1)
			labels_one_hot_train[i][idx] = 1

		# For Validation data
		# Generate word indices
		word1_val_idx = word_idx_matrix(word1_val)
		word2_val_idx = word_idx_matrix(word2_val)

		#Convert to int type
		word1_val_idx = word1_val_idx.astype(int)
		word2_val_idx = word2_val_idx.astype(int)

		#one hot representation of labels
		labels_one_hot_val = np.zeros((len(label_val),num_label))
		# print(num_label)	
		for i in range(len(label_val)):
			idx = int(label_to_idx[label_val[i]] - 1)
			labels_one_hot_val[i][idx] = 1

		# Hyperparameters
		num_batch = int(num_data/batch_size)
		batch_num = 0   #batch number presently considered

		with tf.device('/gpu:0'):		
			while batch_num < (num_batch+1):
				x1_batch = word1_train_idx[batch_num*batch_size : (batch_num + 1)*batch_size if (batch_num + 1)*batch_size < word1_train_idx.shape[0] else word1_train_idx.shape[0]]
				x2_batch = word2_train_idx[batch_num*batch_size : (batch_num + 1)*batch_size if (batch_num + 1)*batch_size < word2_train_idx.shape[0] else word2_train_idx.shape[0]]
				y = labels_one_hot_train[batch_num*batch_size : (batch_num + 1)*batch_size if (batch_num + 1)*batch_size < word1_train_idx.shape[0] else word1_train_idx.shape[0]]

				batch_num += 1

				sess.run(train_step, feed_dict = {X1 : x1_batch, X2 : x2_batch, Y: y })

				# if(i % recording_interval == 0):
				# 	print(accuracy.eval(feed_dict={X1 : x1_batch, X2 : x2_batch, Y : y}))

				# if(i == num_iterations - 1):
				print('accuracy = ', accuracy.eval(feed_dict={X1: word1_val_idx, X2: word2_val_idx, Y: labels_one_hot_val}))# sess.run(accuracy, feed_dict={X1: look_up_1, X2: look_up_2, Y: labels_one_hot}))			

	print("Training Completed\n")
	saver = tf.train.Saver()
	save_path = saver.save(sess,"/home/sid/repo/git_repo/Compound-Noun-Interpretation/weights.cpkt")	
	print("saved_in", save_path)
	print("Saved\n")	

	sess.close()
