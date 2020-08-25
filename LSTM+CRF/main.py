# Libraries import
import sys
import os
import time
import random
import spacy
import torchtext
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from utils import BucketBatchSampler, BucketDataset

sys.path.insert(0, os.path.abspath('../')) # To import conll
from conll import evaluate
from models.lstm import LSTMTagger
#from models.lstm_crf import LSTMCRFTagger

# Data files and output
train_file = open('../dataset/NL2SparQL4NLU.train.conll.txt', 'r')
train_features_file = open('../dataset/NL2SparQL4NLU.train.features.conll.txt', 'r')
test_file = open('../dataset/NL2SparQL4NLU.test.conll.txt', 'r')
test_features_file = open('../dataset/NL2SparQL4NLU.test.features.conll.txt', 'r')
result_file = open('result.txt', 'w')

# Check arguments and set the chosen embedding and model
if(len(sys.argv) < 3):
	print("Wrong usage. Please specify: \n - desired embedding: ['default' | 'spacy' | 'glove'] \n - architecture: ['lstm' | 'lstm_crf']")
	raise SystemExit
embed_chosen = sys.argv[1]
model_chosen = sys.argv[2]

# Try everything for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
torch.manual_seed(42)
	

##################################
# Parameter variables
##################################
# Network parameters
EMBEDDING_DIM = 256	# Used when the chosen embedding is 'default' 
HIDDEN_DIM = 64
bi = True	# Whether it is bidirectional
epochs = 100	# Good amount is 100

# Optimizer parameters
lr = 0.1		# Learning rate
wd = 1e-5	# Weight decay
mmtm = 0.6		# Momentum

# Vector with embedding weights. None in case of 'default' embedding
# Correct values will be assigned in case 'glove' or 'spacy' are chosen
embedding_weights = None

# Other parameters
BATCH = 50	# Batch size
lemma = False	# Whether to use lemmas or not


########################
# Reading of train file
########################
# Sentences of corpus
train_sents = []
tmp_w = []
tmp_c = []
labels = []

# Compute the max length in the meanwhile
max_sent = 0

# Read all sentences and concepts
# Save them in separate lists
for t_line, f_line in zip(train_file, train_features_file):
	a = list(t_line.split()) # Words Tags
	f = list(f_line.split()) # Words POS-Tags Lemma

	if len(a) > 0:

		if lemma:
			a[0] = f[2]
		tmp_w.append(a[0])
		tmp_c.append(a[1])
	else:
		train_sents.append(tmp_w)
		labels.append(tmp_c)
		
		# For the max length
		if len(tmp_w) > max_sent:
			max_sent = len(tmp_w)
		
		tmp_w = []
		tmp_c = []
		
########################
# Reading of test file
########################
# Sentences of corpus
test_sents = []
test_sents_conc = [] # Used for evaluation
tmp_w = []
tmp_wc = []

# Read all sentences and concepts
# Save them in separate lists
for t_line, f_line in zip(test_file, test_features_file):
	a = list(t_line.split()) # Words Tags
	f = list(f_line.split()) # Words POS-Tags Lemma

	if len(a) > 0:
		if lemma:
			a[0] = f[2]
		tmp_w.append(a[0])
		tmp_wc.append(tuple(a))
	else:
		test_sents.append(tmp_w)
		test_sents_conc.append(tmp_wc)
		tmp_w = []
		tmp_wc = []


#####################################################
# Vocabulary for the concepts can be always the same
#####################################################
# Generate concepts vocabulary
# Sorted for reproducibility
conc_set = sorted(set([conc for sent in labels for conc in sent]))
conc_vocab = {}
for conc in conc_set:
	conc_vocab[conc] = len(conc_vocab)
	
def prepare_tags(sent, vocab):
	seq = [vocab[w] for w in sent]
		
	return torch.tensor(seq, dtype=torch.long)


###########################################################################
# Prepare sequence will be always the same but with different vocabularies
###########################################################################
def prepare_sequence(sent, vocab):
		
		seq = []
		for w in sent:
			if w in vocab:
				seq.append(vocab[w])
			else:
				seq.append(vocab['<unk>'])
			
		return torch.tensor(seq, dtype=torch.long)


##############################
# Word vocabularies if custom
##############################
if embed_chosen == 'default':
	# Generate a word set
	# Sorted for reproducibility
	word_set = sorted(set([word for sent in train_sents for word in sent]))

	# Generate vocabulary
	word2idx = {}
	for w in word_set:
		word2idx[w] = len(word2idx)

	# Add padding element and 'unk' element:
	word2idx['<pad>'] = len(word2idx)
	word2idx['<unk>'] = len(word2idx)


############################
# Word vocabularies if spacy
############################
if embed_chosen == 'spacy':
	# Load spacy's largest model
	nlp = spacy.load("en_core_web_lg")

	print(dir(nlp))


############################
# Word vocabularies if glove
############################
if embed_chosen == 'glove':

	# Load smallest glove embedding. The first time it will download the data
	glove = torchtext.vocab.GloVe(name='6B', dim=300)

	# Change embedding size to match glove's vectors
	EMBEDDING_DIM = glove.dim

	# This is a matrix which contains all the embedding vectors we'll use as weights
	embedding_weights = glove.vectors
	
	# This is a dictionary word to index, with indexes corresponding to the right glove vectors
	word2idx = glove.stoi
	# Default value for unknown words for glove, not present when assigning stoi
	word2idx['<unk>'] = 0


#############################################################
# Define custom function to check the F1 score of train/test
#############################################################
# Structure of the thesis, hypothesis:
# thesis = [[('I', 'O'), ('am', 'O'), ('fierce', 'I-actor.name')]]
# hypothesis = [[('I', 'O'), ('am', 'O'), ('fierce', 'I-actor.name')]]

def train_evaluation(predictions):
	
	# We have tuples (sentence, labels, pred_labels)
	# Transform those in the right format (see above)
	correct = []
	predicted = []
	
	for el in predictions:
		s = el[0] # It doesn't matter if it's encoded as long as tokens are recognizable
		lab = [list(conc_vocab.keys())[list(conc_vocab.values()).index(idx)] for idx in el[1]]
		pred_lab = conc_from_score(el[2])
		
		# Tmp lists to store the sentences
		tmp_c = []
		tmp_p = []
		
		for w, l, t in zip(s,lab, pred_lab):
			# Append (word, label) for correct and (word, predicted) for 
			tmp_c.append(tuple([w, l]))
			tmp_p.append(tuple([w, t]))
			
		# Finally append the reconstructed sentences
		correct.append(tmp_c)
		predicted.append(tmp_p)
		
	# Now compute the evaluation using conll.py
	return evaluate(correct, predicted)
	
def test_evaluation(predictions):
	
	preds = []	
	for sent, t_sent in zip(test_sents, predictions):
		preds.append([tuple([a, b]) for a, b in zip(sent, t_sent)])

	# Now compute the evaluation F1 score using conll.py
	return evaluate(test_sents_conc, preds)


###########################################
# Function to get concepts from the scores
###########################################
def conc_from_score(scores):
	
	# List to keep the index of the maximum score
	idx_scores = [s.index(max(s)) for s in scores.tolist()]

	# Return the concepts mapped to these indexes
	return [list(conc_vocab.keys())[list(conc_vocab.values()).index(idx)] for idx in idx_scores]
	
	
#############################
# Prepare train data batches
#############################
# The technique used is to have batches with sentences which have the same length
# The maximum length is pretty short, so it is very likely to have sentences with similar length

# First vectorize the sequences
train_vec = [prepare_sequence(sent, word2idx) for sent in train_sents]
labels_vec = [prepare_tags(sent, conc_vocab) for sent in labels]

# Now instantiate the bucket sampler and bucket  
bucket_batch_sampler = BucketBatchSampler(BATCH, train_vec, labels_vec)
bucket_dataset = BucketDataset(train_vec, labels_vec)

# Now use the data loader for the iterator over batches
data_batch = DataLoader(bucket_dataset, batch_sampler=bucket_batch_sampler, shuffle=False, drop_last=False, num_workers=0)
batches_number = data_batch.batch_sampler.batch_count()
print("Number of batches: ", batches_number)


####################
# Instantiate model
####################
# Choose the right model 
if model_chosen == 'lstm':
	model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(conc_vocab), embed_chosen, bi, embedding_weights)
elif model_chosen == 'lstm+crf':
	model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(conc_vocab), embed_chosen, bi, embedding_weights)

loss_function = torch.nn.NLLLoss()	# NLL is used for multi classification tasks
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mmtm)


########################
# Training of the model
########################

# Save losses and F1 scores
history = defaultdict(list)

# Training loop
print("=================================================================================")
print("Training started")
model.train()
for epoch in range(1, epochs+1):
	# To keep track of the loss and F1 scores for the epoch
	loss_sum = 0
	f1 = 0
	
	# Keep track of the predicted tags for easier F1 computation
	tags_sum = []
	
	t0 = time.time()
	i = 1
	for batch_in, batch_out in data_batch:
		sent_len = [len(a) for a in batch_out]	
		
		# Clear gradients
		model.zero_grad()

		# Run forward pass
		tag_scores = model(batch_in)
		
		# Save for F1 evaluation. Reshape is for separating the sentences
		# We have to save the original labels as well, to maintain the order in the batches
		reshaped = tag_scores.reshape(len(batch_in), sent_len[0], len(conc_vocab))
		tags_sum.extend([(sent, labels, tag_list) for sent, labels, tag_list in zip(batch_in, batch_out, reshaped)])
		
		# Compute loss, gradients, and update parameters by calling optimizer.step()
		loss = loss_function(tag_scores, torch.flatten(batch_out))
		loss.backward()
		optimizer.step()
		
		loss_sum += loss.item()
	
	# Try on test to check the F1 evolution on it
	with torch.no_grad():
		test_preds = []
		
		for sent in test_sents:
			# Prepare the sentence and feed it to the model
			inp = prepare_sequence(sent, word2idx).reshape(1, len(sent))
			
			tag_scores = model(inp)
			
			# Save the tag sequence
			test_preds.append(conc_from_score(tag_scores))
			
		# Compute evaluation
		ev = test_evaluation(test_preds)['total']['f']
		history['f1_test'].append(ev)
	
	t1 = time.time()
	
	# Print some info at the end of the epoch #total
	e_loss = loss_sum / batches_number
	ev_train = train_evaluation(tags_sum)
	f1 = ev_train['total']['f']
	print(f"Epoch {epoch}: Loss = {e_loss:.4f}\t F1 = {f1:.4f}\t Test F1 = {ev:.4f}\t Time = {t1-t0:.4f}s")
	
	# Save info
	history['loss'].append(e_loss)
	history['f1'].append(f1)
	
# Print out the results of the training
print("=================================================================================")
print("Training results")
pd_tbl = pd.DataFrame().from_dict(ev_train, orient='index')
print(pd_tbl)
print("=================================================================================")


###########################################
# Plot the learning curve and the f1 score
###########################################
plt.plot(history['loss'])
plt.plot(history['f1'])
plt.plot(history['f1_test'])
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.title('Loss and F1 Through Epochs')
plt.legend(['NLLLoss', 'F1 Score Train', 'F1 Score Test'])
plt.savefig('learning_curve_{}.png'.format(model_chosen))
plt.show()
	
	
####################
# Tag the test set
####################
print("Tagging test set")
model.eval()
test_preds = []

for sent in test_sents:
	# Prepare the sentence and feed it to the model
	inp = prepare_sequence(sent, word2idx).reshape(1, len(sent))
	tag_scores = model(inp)

	# Save the tag sequence
	test_preds.append(conc_from_score(tag_scores))

# Compute evaluation
ev = test_evaluation(test_preds)

# Print out a good formatted evaluation
print("Tagging and test evaluation complete")
print("Evaluation results:")
pd_tbl = pd.DataFrame().from_dict(ev, orient='index')
print(pd_tbl)
pd_tbl.to_csv('evaluation_python.txt')
print("=================================================================================")


##########################################
# Generate output file with computed tags
##########################################
for sent, tagged in zip(test_sents, test_preds):
	for i in range(len(sent)):
		string = sent[i] + '\t' + tagged[i] + '\n'
		result_file.write(string)
	result_file.write('\n')
print("Result file written")
