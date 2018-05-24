def read_data():
	#Read labels(or interpretations) and the words
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/tratz_and_hovy_full_relations.txt', "r") as label_list:
	    label = label_list.read().split('\n')	

	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/tratz_and_hovy_full.dict', "r") as word_list:
	    word = word_list.read().split('\n')

	# for word in words:
	# 	print(word)

	# Keep labels and words in the form of dictionaries
	count = 0
	idx_to_label = {}
	label_to_idx = {}
	for i in label:
		count += 1
		idx_to_label[count] = i
		label_to_idx[i] = count
	
	# print(labels)

	count = 0    
	idx_to_word = {}
	word_to_idx = {}

	for i in word:
		count += 1
		word_to_idx[i] = count
		idx_to_word[count] = i 

	# print(words)	

	return word_to_idx, idx_to_word,label_to_idx,idx_to_label

def read_train_data(words,labels):   #(idx_to_word, idx_to_label)
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/dataset/tratz_and_hovy_full_train.dataset', "r") as row_data:
		row = row_data.read().split('\n')

	count = 0
	word_1 = []
	word_2 = []
	label = []

	for i in row :
		count +=1
		# if(count > 200):
		# 	break

		if i == '':   #Check if file is at EOF
			break

		# print(count)	
		
		#Split each row into individual numbers
		i = i.split(' ')

		#store words at corresponding index of training data
		# word_1[count] = words[int(i[0])]     #int(i[0]) - int typecast as word accepts int type data
		# word_1[words[int(i[0])]] = count     #int(i[0]) - int typecast as word accepts int type data
		# word_2[words[int(i[1])]] = count
		# label[labels[int(i[2])]] = 	count
		word_1.append(words[int(i[0])])
		word_2.append(words[int(i[1])])
		label.append(labels[int(i[2])])


	# print(label)	

	return word_1, word_2, label

def read_test_data(words,labels):  #(idx_to_word, idx_to_label)
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/dataset/tratz_and_hovy_full_train.dataset', "r") as row_data:
		row = row_data.read().split('\n')

	count = 0
	word_1 = []
	word_2 = []
	label = []

	for i in row :
		count +=1
		# if(count > 200):
		# 	break


		if i == '':   #Check if file is at EOF
			break

		# print(count)	
		
		#Split each row into individual numbers
		i = i.split(' ')

		#store words at corresponding index of training data
		# word_1[count] = words[int(i[0])]     #int(i[0]) - int typecast as word accepts int type data
		# word_1[words[int(i[0])]] = count     #int(i[0]) - int typecast as word accepts int type data
		# word_2[words[int(i[1])]] = count
		# label[labels[int(i[2])]] = 	count
		word_1.append(words[int(i[0])])
		word_2.append(words[int(i[1])])
		label.append(labels[int(i[2])])


	# print(label)	

	return word_1, word_2, label	
		
if __name__ == "__main__":
	word_to_idx, idx_to_word,label_to_idx, idx_to_label = read_data()
	word1, word2, label = read_train_data(idx_to_word,idx_to_label)		
	print(word1[9])