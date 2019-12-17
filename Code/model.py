import torch
import torch.nn as nn

## Two Models will be trained LSTM and Deep Neural
class LSTMmodel(nn.Module) :

	def __init__(self, input, hidden_units, seq_len, pred_len) :

		super(LSTMmodel, self).__init__()
		## Batch First ensures input to be of shape (batch, seq, input)
		self.lstm1 = nn.LSTM(input, hidden_units, num_layers=1, batch_first=True)
		self.lstm2 = nn.LSTM(hidden_units, hidden_units, num_layers=1, batch_first=True)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.dense1 = nn.Linear(in_features=hidden_units*seq_len, out_features=128)
		self.dense2 = nn.Linear(in_features=128, out_features=pred_len)

		self.input = input
		self.hidden_units = hidden_units
		self.seq_len = seq_len
		self.pred_len = pred_len
		#init_hidden()

	def init_hidden(self, batch_size):

		self.batch_size = batch_size
		self.hidden_state1 = (torch.zeros(1, self.batch_size, self.hidden_units), torch.zeros(1, self.batch_size, self.hidden_units))
		self.hidden_state2 = (torch.zeros(1, self.batch_size, self.hidden_units), torch.zeros(1, self.batch_size, self.hidden_units))

	def forward(self, x) :

		## x is of shape (batch, dims)
		x = x.reshape(self.batch_size, self.seq_len, -1)
		## x of shape (batch, seq, input)
		out1, self.hidden_state1 = self.lstm1(x, self.hidden_state1)
		out1 = self.relu(out1)
		out2, self.hidden_state2 = self.lstm2(out1, self.hidden_state2)
		out2 = self.relu(out2)
		#print(out2.shape, self.batch_size)
		out3 = self.dense1(out2.reshape(self.batch_size, -1))
		out3 = self.relu(out3)
		out4 = self.dense2(out3)
		#arous = self.dense2(out2.reshape(self.batch_size, -1))
		## Sets output to value -1, 1
		result = self.tanh(out4)
		#temp = temp.unsqueeze(dim=2)
		#humid = humid.unsqueeze(dim=2)

		return result
        
## Model for Densely connected Neural Networks
class NeuralModel(nn.Module) :

	def __init__(self, input) :
	
		super(NeuralModel, self).__init__()
		self.dense1 = nn.Linear(in_features = input, out_features = 128)
		self.dense2 = nn.Linear(in_features = 128, out_features = 16)
		self.dense3 = nn.Linear(in_features = 16, out_features = 2)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		
	def forward(self, x) :
		
		out1 = self.dense1(x)
		out1 = self.relu(out1)
		out2 = self.dense2(out1)
		out2 = self.relu(out2)
		out3 = self.dense3(out2)
		## Tanh activation for values from -1 to 1
		result = self.tanh(out3)
		
		return result
		
		
