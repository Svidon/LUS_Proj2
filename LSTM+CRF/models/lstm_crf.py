# Import libraries
from torch import nn
from torch.nn import functional as F
from torchcrf import CRF

class LSTMCRFTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional, emb_weights):
		super(LSTMCRFTagger, self).__init__()

		self.tagset_size = tagset_size

		###########################
		# Embeds handling
		###########################
		if emb_weights is not None:
			self.word_embeddings = nn.Embedding.from_pretrained(emb_weights, freeze=True)
		else:
			self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		# The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // (1 if not bidirectional else 2),
			num_layers=1, bidirectional=bidirectional, batch_first=True)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

		# CRF layer
		self.crf = CRF(tagset_size, batch_first=True)


	def forward(self, sentences, tags):

		# Compute the sentence length (all equal in the batch)
		sent_len = [len(a) for a in sentences][0]

		# No need to take in account the batch dimension
		embeds = self.word_embeddings(sentences)
		
		# Batch first is true so we don't need to work on dimensions
		lstm_out, _ = self.lstm(embeds) # Dimension is [len(sentences), len_sent, -1]
		
		# Flatten the LSTM output and feed it to the linear layer
		linear_input = lstm_out.contiguous().view(-1, lstm_out.shape[2])
		tag_space = self.hidden2tag(linear_input)

		# Reshape the vector to feed it to the CRF
		tag_reshaped = tag_space.reshape(len(sentences), sent_len, self.tagset_size)

		# Return the mean NLL over the batch
		return -self.crf(tag_reshaped, tags, reduction='mean')

	def predict(self, sentences):

		# Compute the sentence length (all equal in the batch)
		sent_len = [len(a) for a in sentences][0]

		# No need to take in account the batch dimension
		embeds = self.word_embeddings(sentences)
		
		# Batch first is true so we don't need to work on dimensions
		lstm_out, _ = self.lstm(embeds) # Dimension is [len(sentences), len_sent, -1]
		
		# Flatten the LSTM output and feed it to the linear layer
		linear_input = lstm_out.contiguous().view(-1, lstm_out.shape[2])
		tag_space = self.hidden2tag(linear_input)

		# Reshape the vector to feed it to the CRF
		tag_reshaped = tag_space.reshape(len(sentences), sent_len, self.tagset_size)

		# Return the predicted sequence
		return self.crf.decode(tag_reshaped)