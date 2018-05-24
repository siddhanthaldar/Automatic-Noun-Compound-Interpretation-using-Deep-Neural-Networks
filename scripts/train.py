import tensorflow as tf
import numpy as np
import read
import embedding

def look_up_table(embedding_matrix, word, embedding_dim):  #embedding_matrix : numpy array, word : list
	
	look_up = np.zeros((len(word), embedding_dim))
	count = 0

	for i in word:
		idx = word_to_idx[i]
		look_up[count] = embedding_matrix[idx-1]
		count += 1

	return look_up	


word_to_idx, idx_to_word,label_to_idx,idx_to_label = read.read_data()
word1, word2 ,train_label = read.read_train_data(idx_to_word, idx_to_label)
embedding_matrix = embedding.glove_embedding(idx_to_word)

num_data = len(word1)
embedding_dim = embedding_matrix.shape[1]
num_label = len(label_to_idx) - 1

# print(num_label)

#look up for each word
look_up_1 = np.zeros((len(word1), embedding_dim))
look_up_2 = np.zeros((len(word2), embedding_dim))
look_up_1 = look_up_table(embedding_matrix, word1, embedding_dim)
look_up_2 = look_up_table(embedding_matrix, word2, embedding_dim)

# print(word1[0])
print(look_up_1.shape)

#one hot representation of labels
labels_one_hot = np.zeros((len(train_label),num_label))
# print(num_label)	
for i in range(len(train_label)):
	idx = int(label_to_idx[train_label[i]] - 1)
	labels_one_hot[i][idx] = 1

# print(labels_one_hot[3])


# ***************** Model definition ************************

#*************** PyTorch **************************************
# print(look_up_1.shape)
# hidden_1 = torch.nn.Linear(embedding_dim,embedding_dim)(look_up_1)
# hidden_2 = torch.nn.Linear(embedding_dim,embedding_dim)(look_up_2)
# hidden = hidden_1 + hidden_2
# hidden = F.relu(hidden)

# output = torch.nn.Linear(embedding_dim, num_label)(hidden)
# output = softmax(output)


# **********************TensorFlow*****************************

#Necessary Functions
def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def mul_mat(X,W,b):
	return tf.matmul(X,W) + b

X1 = tf.placeholder(tf.float32, shape = ([None, embedding_dim]))
X2 = tf.placeholder(tf.float32, shape = ([None, embedding_dim]))
Y = tf.placeholder(tf.float32, shape = ([None, num_label]))  #labels

# print(look_up_1.shape)
# word_1 = tf.Variable(tf.constant(look_up_1))
# word_2 = tf.Variable(tf.constant(look_up_2))  #tf.get_variable('word_2', initializer = tf.constant(look_up_2))
# y = tf.Variable(tf.constant(labels_one_hot))
# label_model = tf.Variable(tf.constant())
# print(word_1)

#hidden layer
W_h1 = weight_variable((embedding_dim, embedding_dim))
b_h1 = bias_variable([1, embedding_dim])
W_h2 = weight_variable((embedding_dim, embedding_dim))
b_h2 = bias_variable([1, embedding_dim])

h = tf.nn.relu(mul_mat(X1,W_h1, b_h1) + mul_mat(X2,W_h2, b_h2))

# output layer
W_out = weight_variable([embedding_dim, num_label])
b_out = bias_variable([1, num_label])

output = mul_mat(h,W_out,b_out)   #tf.nn.softmax(mul_mat(h,W_out,b_out))
	
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = output))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

num_iterations = 500000
recording_interval = 10000
batch_size = 500
num_batch = int(num_data/batch_size)
batch_num = 0   #batch number presently considered

init = tf.global_variables_initializer()

#Train
with tf.Session() as sess:
	sess.run(init)	
	with tf.device('/gpu:0'):
		for i in range(num_iterations):
			x1_batch = look_up_1[batch_num*batch_size : (batch_num + 1)*batch_size - 1]
			x2_batch = look_up_2[batch_num*batch_size : (batch_num + 1)*batch_size - 1]
			y = labels_one_hot[batch_num*batch_size : (batch_num + 1)*batch_size - 1]

			batch_num += 1
			if(batch_num >= num_batch):
				batch_num = 0

			sess.run(train_step, feed_dict = {X1 : x1_batch, X2 : x2_batch, Y: y })

			if(i % recording_interval == 0):
				print(accuracy.eval(feed_dict={X1 : x1_batch, X2 : x2_batch, Y : y}))

			if(i == num_iterations - 1):
				print('accuracy = ', accuracy.eval(feed_dict={X1: look_up_1, X2: look_up_2, Y: labels_one_hot}))# sess.run(accuracy, feed_dict={X1: look_up_1, X2: look_up_2, Y: labels_one_hot}))			

	print("Training Completed\n")
	saver = tf.train.Saver([W_h1, b_h1, W_h2, b_h2, W_out, b_out])
	save_path = saver.save(sess,"/home/sid/virtual_env/nlp/proj/Compound-Noun-Interpretation/wgts_glove300d.cpkt")	
	print("saved_in", save_path)
	print("Saved\n")	

	sess.close()




