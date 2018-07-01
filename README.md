# Compound-Noun-Interpretation
This is an TensorFlow implementation of the paper on [Automatic Noun Compound Interpretation using
Deep Neural Networks and Word Embeddings](http://www.sfs.uni-tuebingen.de/~cdima/papers/IWCS201522.pdf).
This paper focuses on identifying the semantic relation that holds between the constituents of a compound.
The code is divided into several modules and has a version that uses fine tuning and one that does not. The instructions to run the package is given further.

## Code Description
The python scripts are present in the ***scripts*** folder. The scripts contain :

- **read.py** : for reading the train, test and validation data
- **embedding.py** - prepares the embedding matrix for different word embeddings
- **train.py** : train the data for a chosen word embedding type. The instruction for choosing different word embedding is given below.
- **test.py** : test accuracy for a chosen word embedding type.
- **train_with_f.py** : train the data with fine tuning of the embedding matrix for a chosen word embedding type. The instruction for choosing different word embedding is given below. 
- **test_with_f.py** : test accuracy for the data trained using fine tuning for a chosen word embedding type.

## Instructions
- **Train without fine tuning** : Modify the script ***train.py***. Set the type of word embedding by changing the function name in line 22 of the given script. The function names for different embedding types can be seen from ***embedding.py***. The same must be done to ***test.py*** for obtaining accuracy on the test set.
- **Train with fine tuning** : Modify the script ***train_with_f.py***. Set the type of word embedding by changing the function name in line 20 of the given script. The function names for different embedding types can be seen from ***embedding.py***. The same must be done to ***test_with_f.py*** for obtaining accuracy on the test set.



