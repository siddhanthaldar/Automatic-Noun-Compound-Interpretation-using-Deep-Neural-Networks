def read_data():
	#Read labels(or interpretations) and the words
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/tratz_and_hovy_full_relations.txt', "r") as label_list:
	    label = label_list.read().split('\n')	

	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/tratz_and_hovy_full.dict', "r") as word_list:
	    word = word_list.read().split('\n')

	# Keep labels and words in the form of dictionaries
	count = 0
	idx_to_label = {}
	label_to_idx = {}
	for i in label:
		count += 1
		idx_to_label[count] = i
		label_to_idx[i] = count
	
	count = 0    
	idx_to_word = {}
	word_to_idx = {}

	for i in word:
		count += 1
		word_to_idx[i] = count
		idx_to_word[count] = i 

	return word_to_idx, idx_to_word,label_to_idx,idx_to_label

def read_train_data(words,labels):   #in the form (idx_to_word, idx_to_label)
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/dataset/tratz_and_hovy_full_train.dataset', "r") as row_data:
		row = row_data.read().split('\n')

	word_1 = []
	word_2 = []
	label = []

	for i in row :
		if i == '':   #Check if file is at EOF
			break
		
		#Split each row into individual numbers
		i = i.split(' ')

		#store words at corresponding index of training data
		word_1.append(words[int(i[0])])
		word_2.append(words[int(i[1])])
		label.append(labels[int(i[2])])

	return word_1, word_2, label

def read_test_data(words,labels):  # in the form (idx_to_word, idx_to_label)
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/dataset/tratz_and_hovy_full_test.dataset', "r") as row_data:
		row = row_data.read().split('\n')

	word_1 = []
	word_2 = []
	label = []

	for i in row :
		if i == '':   #Check if file is at EOF
			break

		#Split each row into individual numbers
		i = i.split(' ')

		#store words at corresponding index of training data
		word_1.append(words[int(i[0])])
		word_2.append(words[int(i[1])])
		label.append(labels[int(i[2])])

	return word_1, word_2, label	

def read_val_data(words,labels):   # in the form (idx_to_word, idx_to_label)
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/dataset/tratz_and_hovy_full_dev.dataset', "r") as row_data:
		row = row_data.read().split('\n')

	word_1 = []
	word_2 = []
	label = []

	for i in row :
		if i == '':   #Check if file is at EOF
			break
		
		#Split each row into individual numbers
		i = i.split(' ')

		#store words at corresponding index of training data
		word_1.append(words[int(i[0])])
		word_2.append(words[int(i[1])])
		label.append(labels[int(i[2])])

	return word_1, word_2, label

def read_cross_val_data(words,labels):   # in the form (idx_to_word, idx_to_label)
	word_1_total = []
	word_2_total = []
	label_total = []

	for x in range(10):

		with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/dataset/10-fold-train_dev/tratz_and_hovy_full_train_and_dev_cv' + str(x+1) + '.dataset', "r") as row_data:
			row = row_data.read().split('\n')

		word_1 = []
		word_2 = []
		label = []

		for i in row :
			if i == '':   #Check if file is at EOF
				break

			#Split each row into individual numbers
			i = i.split(' ')

			#store words at corresponding index of training data
			word_1.append(words[int(i[0])])
			word_2.append(words[int(i[1])])
			label.append(labels[int(i[2])])

		word_1_total.append(word_1)
		word_2_total.append(word_2)
		label_total.append(label)	

	return word_1_total, word_2_total, label_total	

		
if __name__ == "__main__":
	word_to_idx, idx_to_word,label_to_idx, idx_to_label = read_data()
	word1, word2, label = read_cross_val_data(idx_to_word,idx_to_label)		
	print(len(word1[1]))