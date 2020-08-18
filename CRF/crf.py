import os
import sys
import scipy
import pandas as pd

sys.path.insert(0, os.path.abspath('../')) # To import conll
from conll import evaluate
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Read train and test files
train = open('../dataset/NL2SparQL4NLU.train.conll.txt', 'r')
train_features = open('../dataset/NL2SparQL4NLU.train.features.conll.txt', 'r')
test = open('../dataset/NL2SparQL4NLU.test.conll.txt', 'r')
test_features = open('../dataset/NL2SparQL4NLU.test.features.conll.txt', 'r')

# Window of words for the features (symmetric)
WIN = 5
print("WINDOW: ", WIN)

########################
# Reading of train file
########################

# Sentences of corpus
train_sents = []
tmp = []
concepts = []

# Identify all sentences with tuples of word-tag
for t_line, f_line in zip(train, train_features):
	a = list(t_line.split()) # Words Tags
	f = list(f_line.split()) # Words POS-Tags Lemma

	if len(a) > 0:
		# Contruct a tuple with: word, lemma, POS, concept
		a = [a[0], f[2], f[1], a[1]]
		tmp.append(tuple(a))
		concepts.append(a[3])
	else:
		train_sents.append(tmp)
		tmp = []
		
# Unique concepts and remove 'O' for the evaluation of F1 (wanna optimize on the others)
concepts = list(set(concepts))
concepts.remove('O')

########################
# Reading of test file
########################

# Sentences of corpus
test_sents = []
tmp = []

# Identify all sentences with tuples of word-tag
for t_line, f_line in zip(test, test_features):
	a = list(t_line.split()) # Words Tags
	f = list(f_line.split()) # Words POS-Tags Lemma

	if len(a) > 0:
		# Contruct a tuple with: word, lemma, POS, concept
		a = [a[0], f[2], f[1], a[1]]
		tmp.append(tuple(a))
	else:
		test_sents.append(tmp)
		tmp = []
		
		
################################
# Create the features template
################################
def word2features(sent, i, window):
	word = sent[i][0]
	lemma = sent[i][1]
	postag = sent[i][2]
    
	features = {
		'bias': 1.0,				# Default bias to add
		'word': word,				# All the words are already lowkey
		'word[:2]': word[:2],			# Prefix
		'word[-3:]': word[-3:],		# Suffix
		'word.isdigit()': word.isdigit(),	# Whether the word is a digit number (there are some)
		'lemma': lemma,			# Lemma corresponding to the word
		'postag': postag,			# Postag
		'postag[:2]': postag[:2]        	# Prefix of postag (other values don't make too much sense)
	}
	
	# If it is not the first word of the sentence get the features of the previous words (like ngrams)
	# Else add a feature meaning it's the fist word of the sentence
	if i > 0:
		# Check all the words inside the window
		for k in range(1, window+1):
			# Check not to be in negative indexes
			if (i-k) >= 0:			
				word1 = sent[i-k][0]
				lemma1 = sent[i-k][1]
				postag1 = sent[i-k][2]
				features.update({
				    '-{}:word'.format(k): word1,
				    '-{}:word.isdigit()'.format(k): word1.isdigit(),
				    '-{}:postag'.format(k): postag1
				})
	else:
		features['BOS'] = True

	# If it is not the last word of the sentence get the features of the following words (future information)
	# Else add a feature meaning it's the last word of the sentence
	if i < len(sent)-1:
		# Check all the words inside the window
		for k in range(1, window+1):
			# Check that the index doesn't exceed the sentence length
			if (i+k) <= len(sent)-1:
				word1 = sent[i+k][0]
				lemma1 = sent[i+k][1]
				postag1 = sent[i+k][2]
				features.update({
				    '+{}:word'.format(k): word1,
				    '+{}:word.isdigit()'.format(k): word1.isdigit(),
				    '+{}:postag'.format(k): postag1
				})
	else:
		features['EOS'] = True
		
	return features


###############################################
# Other wrapper functions to create features
# These will be used also for the tokenization
###############################################
# Window is the parameter to control how many previous/future words you want to check
# It is symmetric, so the interval is: [-window,window]
def sent2features(sent, window):
	return [word2features(sent, i, window) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, lemma, pos, label in sent]
    
def sent2tokens(sent):
	return [token for token, lemma, pos, label in sent]


##################################
# Feature extraction
# Both for training and test sets
##################################
# Features for test and train
train_feats = [sent2features(s, WIN) for s in train_sents]
test_feats = [sent2features(s, WIN) for s in test_sents]

# Train labels
train_labels = [sent2labels(s) for s in train_sents]


###############################
# Instantiate model and train
###############################
# Base model (parameters are actually not needed apart from 'all_possible_transitions'
crf = CRF(
	all_possible_transitions=True)

# Hyperparameters on which to look for the optimal
# Two different because the two tried algorithms have different parameters
param_space1 = {
	'algorithm': ['lbfgs'],
	'min_freq': [1,2,3], # Frequency cutoff
	'c1': scipy.stats.expon(scale=0.5),
	'c2': scipy.stats.expon(scale=0.05),
	'max_iterations': [100,1000,2000]}
param_space2 = {
	'algorithm': ['l2sgd'],
	'min_freq': [1,2,3],
	'c2': scipy.stats.expon(scale=0.05),
	'max_iterations': [100,500,1000]} # 1000 is the max for l2sgd

# Prepare a scorer for the parameters search
f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=concepts)

# Make the parameters search
rs_lbfgs = RandomizedSearchCV(crf, param_space1, n_jobs=-1, random_state=1, verbose=1, cv=3, n_iter=20, scoring=f1_scorer)
rs_lbfgs.fit(train_feats, train_labels)

rs_l2sgd = RandomizedSearchCV(crf, param_space2, n_jobs=-1, random_state=1, verbose=1, cv=3, n_iter=20, scoring=f1_scorer)
rs_l2sgd.fit(train_feats, train_labels)

if rs_lbfgs.best_score_ > rs_l2sgd.best_score_:
	# Print the best parameters
	best_params = rs_lbfgs.best_params_
	best_score = rs_lbfgs.best_score_

	print('best params:', best_params)
	print('best CV score:', best_score)

	# Assign to CRF best parameters
	crf = rs_lbfgs.best_estimator_
else:
	# Print the best parameters
	best_params = rs_l2sgd.best_params_
	best_score = rs_l2sgd.best_score_

	print('best params:', best_params)
	print('best CV score:', best_score)

	# Assign to CRF best parameters
	crf = rs_l2sgd.best_estimator_

# Train with the best parameters
crf.fit(train_feats, train_labels)


#######################
# Predict the concepts
#######################
pred = crf.predict(test_feats)
pred_concepts = [[(test_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]

#####################################
# Generate the resulting output file
#####################################
with open('result.txt', 'w') as result:
	# Navigate test sentences and predicted tag lists together
	for sent, tagged in zip(test_sents, pred):
		i = 0
		for word, lemma, pos, concept in sent:
			string = word + '\t' + tagged[i] + '\n'
			result.write(string)
			i += 1
		result.write('\n')


###################################
# Evaluate results and print table
###################################
results = evaluate(test_sents, pred_concepts)

pd_tbl = pd.DataFrame().from_dict(results, orient='index')
print(pd_tbl)
pd_tbl.to_csv('evaluation_window{}_python.txt'.format(WIN))

with open('evaluation_window{}_python.txt'.format(WIN), 'a') as res:
	res.write('\n')
	res.write('Window: {}\n\n'.format(WIN))
	res.write('Best params: {}\n'.format(str(best_params)))
	res.write('Best CV score: {}'.format(str(best_score)))
