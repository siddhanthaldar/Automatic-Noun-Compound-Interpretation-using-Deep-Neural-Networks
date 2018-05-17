def read():
	#Read labels(or interpretations) and the words
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/tratz_and_hovy_full_relations.txt', "r") as label_list:
	    label = label_list.read().split('\n')	

	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/tratz_and_hovy_full.dict', "r") as word_list:
	    word = word_list.read().split('\n')

	# for word in words:
	# 	print(word)

	# Keep labels and words in the form of dictionaries
	count = 0
	labels = {}
	for i in label:
		count += 1
		labels[count] = i
	
	# print(labels)

	count = 0    
	words = {}
	for i in word:
		count += 1
		words[count] = i

	# print(words)	

	return words,labels

def read_train_data(words,labels):
	with open('/home/sid/virtual_env/nlp/proj/dataset/tratz_hovy_full/dataset/tratz_and_hovy_full_train.dataset', "r") as row_data:
		row = row_data.read().split('\n')

	count = 0
	word_1 = {}
	word_2 = {}
	label = {}

	for i in row :
		count +=1
		if i == '':   #Check if file is at EOF
			break
		
		#Split each row into individual numbers
		i = i.split(' ')

		#store words at corresponding index of training data
		word_1[count] = words[int(i[0])]     #int(i[0]) - int typecast as word accepts int type data
		word_2[count] = words[int(i[1])]
		label[count] = labels[int(i[2])]	

	return word_1, word_2, label

if __name__ == "__main__":
	words,labels = read()
	word1, word2, label = read_train_data(words,labels)		    