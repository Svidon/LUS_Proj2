# Libraries import
import sys
import os
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.abspath('../')) # To import conll
from conll import evaluate
from models.lstm import LSTMTagger

# Data files and output
train_file = open('../dataset/NL2SparQL4NLU.train.conll.txt', 'r')
train_features_file = open('../dataset/NL2SparQL4NLU.train.features.conll.txt', 'r')
test_file = open('../dataset/NL2SparQL4NLU.test.conll.txt', 'r')
test_features_file = open('../dataset/NL2SparQL4NLU.test.features.conll.txt', 'r')

# Check arguments and set the chosen embedding and model
if(len(sys.argv) < 3):
	print("Wrong usage. Please specify: \n - desired embedding: ['custom' | 'spacy' | 'glove'] \n - architecture: ['lstm' | 'lstm_crf']")
	raise SystemExit
embed_chosen = sys.argv[1]
model_chosen = sys.argv[2]


##################################
# Parameter variables
##################################
# Network parameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
bi = False	# Whether it is bidirectional
epochs = 50

# Optimizer parameters
lr = 0.01		# Learning rate
wd = 1e-5	# Weight decay
mmtm = 0.4		# Momentum


########################
# Reading of train file
########################
# Sentences of corpus
train_sents = []
train_sents_conc = []	# This will be used for the F1 score evaluation
tmp_w = []
tmp_c = []
tmp_wc = []
labels = []

# Read all sentences and concepts
# Save them in separate lists
for t_line, f_line in zip(train_file, train_features_file):
	a = list(t_line.split()) # Words Tags
	f = list(f_line.split()) # Words POS-Tags Lemma

	if len(a) > 0:
		tmp_w.append(a[0])	# Use lemmas and not words?
		tmp_c.append(a[1])
		tmp_wc.append(tuple(a))
	else:
		train_sents.append(tmp_w)
		labels.append(tmp_c)
		train_sents_conc.append(tmp_wc)
		tmp_w = []
		tmp_c = []
		tmp_wc = []
		
########################
# Reading of test file
########################
# Sentences of corpus
test_sents = []
test_sents_conc = [] # Used for evaluation
tmp_w = []
tmp_c = []
tmp_wc = []

# Read all sentences and concepts
# Save them in separate lists
for t_line, f_line in zip(test_file, test_features_file):
	a = list(t_line.split()) # Words Tags
	f = list(f_line.split()) # Words POS-Tags Lemma

	if len(a) > 0:
		tmp_w.append(a[0])	# Use lemmas and not words?
		tmp_wc.append(tuple(a))
	else:
		test_sents.append(tmp_w)
		test_sents_conc.append(tmp_wc)
		tmp_w = []
		tmp_wc = []


###############################################
# Vocabulary for the concepts is always the same
###############################################
# Generate concepts vocabulary
conc_set = set([conc for sent in labels for conc in sent])
conc_vocab = {}
for conc in conc_set:
	conc_vocab[conc] = len(conc_vocab)

# Padding element will be super high
#conc_vocab['pad'] = 10000
	
def prepare_tags(sent, vocab, pad=0):
	seq = [vocab[w] for w in sent]
		
	# Apply padding if needed
	while(len(seq) < pad):
		seq.append[10000]
		
	return torch.tensor(seq, dtype=torch.long)


#############################
# Prepare sequence if custom
#############################
if embed_chosen == 'custom':
	# Generate a word set
	word_set = set([word for sent in train_sents for word in sent])

	# Generate vocabulary
	vocab = {}
	for w in word_set:
		vocab[w] = len(vocab)

	# Add padding element and 'unk' element:
	vocab['UNK'] = len(vocab)
	#vocab['PAD'] = len(vocab)
	
	# Function to transform the sentence and PAD it
	def prepare_sequence(sent, vocab, pad=0):
		
		seq = []
		for w in sent:
			if w in vocab:
				seq.append(vocab[w])
			else:
				seq.append(len(vocab)-1) # Corresponding to 'UNK'
		
		# Apply padding if needed
		while(len(seq) < pad):
			seq.append['PAD']
			
		return torch.tensor(seq, dtype=torch.long)
		

############################
# Prepare sequence if spacy
############################


############################
# Prepare sequence if glove
############################


#############################################################
# Define custom function to check the F1 score of train/test
#############################################################
# Structure of the thesis, hypothesis:
# eval_test = [[('I', 'O'), ('am', 'O'), ('fierce', 'I-actor.name')]]
# pred_test = [[('I', 'O'), ('am', 'O'), ('fierce', 'I-actor.name')]]

def train_evaluation(predictions):
	# Tuples (word, tag) for predictions data
	tagged = [conc_from_score(pred) for pred in predictions]
	
	preds = []	
	for sent, t_sent in zip(train_sents, tagged):
		preds.append([tuple([a, b]) for a, b in zip(sent, t_sent)])

	# Now compute the evaluation using conll.py
	return evaluate(train_sents_conc, preds)
	
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


####################
# Instantiate model
####################
# Choose the right model 
if model_chosen == 'lstm':
	model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(conc_vocab), embed_chosen, bi)
elif model_chosen == 'lstm+crf':
	model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(conc_vocab), embed_chosen, bi)

loss_function = torch.nn.NLLLoss()	# NLL is used for multi classification tasks
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mmtm)


########################
# Training of the model
########################
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
	inputs = prepare_sequence(train_sents[0], vocab)
	tag_scores = model(inputs)

# Save losses and F1 scores
history = defaultdict(list)

model.train()
for epoch in range(1, epochs+1):
	# To keep track of the loss and F1 scores for the epoch
	loss_sum = 0
	f1 = 0
	
	# Keep track of the predicted tags for easier F1 computation
	tags_sum = []
	
	t0 = time.time()

	for sentence, tags in zip(train_sents, labels):
		# Clear gradients
		model.zero_grad()

		# Prepare the inputs
		sentence_in = prepare_sequence(sentence, vocab)
		targets = prepare_sequence(tags, conc_vocab)

		# Run forward pass
		tag_scores = model(sentence_in)
		tags_sum.append(tag_scores)

		# Compute loss, gradients, and update parameters by calling optimizer.step()
		loss = loss_function(tag_scores, targets)
		loss.backward()
		optimizer.step()
		
		loss_sum += loss.item()
	
	t1 = time.time()
	
	# Print some info at the end of the epoch #total
	e_loss = loss_sum / len(train_sents)
	f1 = train_evaluation(tags_sum)['total']['f']
	print(f"Epoch {epoch}: loss = {e_loss:.4f}\t F1 = {f1:.4f}\t time = {t1-t0:.4f}s")
	
	# Try on test to check the F1 evolution on it
	with torch.no_grad():
		test_preds = []
		
		for sent in test_sents:
			# Prepare the sentence and feed it to the model
			inp = prepare_sequence(sent, vocab)
			tag_scores = model(inp)

			# Save the tag sequence
			test_preds.append(conc_from_score(tag_scores))
			
		# Compute evaluation
		ev = test_evaluation(test_preds)
		history['f1_test'].append(ev['total']['f'])
	
	# Save info
	history['loss'].append(e_loss)
	history['f1'].append(f1)
	

###########################################
# Plot the learning curve and the f1 score
###########################################
plt.plot(history['loss'])
plt.plot(history['f1'])
plt.plot(history['f1_test'])
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.title('Loss and F1 Through Epochs')
plt.legend(['NLLLoss', 'F1 Score', 'F1 Score Test'])
plt.show()
	
	
####################
# Apply on test set
####################
model.eval()
test_preds = []

for sent in test_sents:
	# Prepare the sentence and feed it to the model
	inp = prepare_sequence(sent, vocab)
	tag_scores = model(inp)

	# Save the tag sequence
	test_preds.append(conc_from_score(tag_scores))

# Compute evaluation
ev = test_evaluation(test_preds)

# Print out a good formatted evaluation
pd_tbl = pd.DataFrame().from_dict(ev, orient='index')
print(pd_tbl)
#pd_tbl.to_csv('evaluation_python.txt')











