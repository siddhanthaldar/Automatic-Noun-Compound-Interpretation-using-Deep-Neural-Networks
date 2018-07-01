This folder contains a dataset of English noun-noun compounds created and annotated by Stephen Tratz, 
and obtained from http://www.isi.edu/publications/licensed-sw/fanseparser/. For details about the annotation, see

* 	Stephen Tratz and Eduard Hovy. 2010. A Taxonomy, Dataset, and Classifier for Automatic Noun Compound Interpretation. In Proceedings of the 48th Annual Meeting of the Association 		for Computational Linguistics. Uppsala, Sweden. 

*	Tratz, S. (2011). Semantically-enriched parsing for natural language understanding. University of Southern California.

The dataset was compiled in it's current form in the context of the SFB 833 A3 project (http://www.uni-tuebingen.de/en/research/core-research/collaborative-research-centers/sfb-833/section-a-context/a3-hinrichs.html).

### Files:
*	dataset file in text form: dataset/tratz_and_hovy_full_dataset.txt (19,158 compounds)

*	dictionary file: tratz_and_hovy_full.dict (5242 unique constituents, lowercased)

*	relations file: tratz_and_hovy_full_relations.txt (37 relations)

*	dataset for experiments:
	 - format: (constituent1_index, constituent2_index, relation_index);
		- constituent indices are obtained via the constituent position in the dictionary file;
		- relation indices are obtained via the relation position in the relation file;
	 - files:
	 	- train dataset: tratz_and_hovy_full_train.dataset (13409 compound instances)
	 	- dev dataset: tratz_and_hovy_full_dev.dataset (1920 compound instances)
	 	- test dataset: tratz_and_hovy_full_test.dataset (3829 compound instances)
	 	- full dataset: tratz_and_hovy_full.dataset (19158 instances)
	 	- 10-fold cross-validation set (train+dev): 10-fold-train_dev/, (15329 compound instances in total)

