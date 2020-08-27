# Import libraries
from torch import nn
from torch.nn import functional as F

class LSTMTagger(nn.Module):

	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional, emb_weights):
		super(LSTMTagger, self).__init__()

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


	def forward(self, sentences):		
		# No need to take in account the batch dimension
		embeds = self.word_embeddings(sentences)
		
		# Batch first is true so we don't need to work on dimensions
		lstm_out, _ = self.lstm(embeds) # Dimension is [len(sentences), len_sent, -1]
		
		# Flatten the LSTM output and feed it to the linear layer
		linear_input = lstm_out.contiguous().view(-1, lstm_out.shape[2])
		tag_space = self.hidden2tag(linear_input)
		
		# Finally apply the softmax to have the probability of the results
		tag_scores = F.log_softmax(tag_space, dim=1)
		
		return tag_scores
