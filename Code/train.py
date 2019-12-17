import numpy as np
import matplotlib.pyplot as plt
import torch
from model import LSTMmodel, NeuralModel
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

def plot_two(y1,y2, string) :
	
	Y1 = y1.numpy()
	Y2 = y2.numpy()
	
	X = np.arange(len(Y1))
	plt.plot(X, Y1, label="Actual "+string)
	plt.plot(X, Y2, label="Predicted "+string)
	plt.xlabel('Frames') 
	# naming the y axis 
	plt.ylabel(string) 
	plt.title("LSTM Network Model")
	# giving a title to my graph  	  
	# show a legend on the plot 
	plt.legend()
	plt.show() 

## CCC value for arousal and valence
def concord_cc2(r1, r2):

	mean_pred = torch.mean((r1 - torch.mean(r1))*(r2 - torch.mean(r2)))
	return (2*mean_pred)/(torch.var(r1) + torch.var(r2) + (torch.mean(r1)- torch.mean(r2))**2)

label_file = open('Input_Faces_Project.csv', 'r')
labels = []

print("Loading Data .......................")

## READ Labels from file
for line in label_file :

	line = line.split(",")
	valence = float(line[1])
	arousal = float(line[2])
	
	labels.append([valence, arousal])
	
labels = np.array(labels)
	
data_file = "Data/bottleneck_"
x = None

## Read Np data of bottleneck features
for index in range(len(labels)) :

	file_path = data_file+str(index)+".npy"
	
	x_data = np.load(file_path)
	#print(x_data.shape)
	if index != 0 :
		x = np.concatenate((x, x_data), axis=0)
	else :
		x = x_data
		
print("Data Load Complete .............", "Data Shape : ", x.shape, "Labels Shape : ", labels.shape)

scaler = StandardScaler()
X = scaler.fit_transform(x)
Y = labels

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

period = 10
max_epoch = 1000
batch_size = 64
seq_len = 16
#loss_prev = np.inf
best_loss = np.inf
train_loss_list = []
test_loss_list = []
train_size = X_train.shape[0]
test_size = X_test.shape[0]

isLSTM = False

if isLSTM :
	## LSTM (input, hidden, seq_len, output)
	framework = LSTMmodel(X_train.shape[1]//16, 35, 16, 2)
else :
	framework = NeuralModel(X_train.shape[1])
optimizer = torch.optim.Adam([{'params' : framework.parameters()}], lr=0.0001)


isTrain = False


if isTrain == True :
	print("Training Start .....................\n")
	for epoch in range(max_epoch):
		
		perm = np.random.permutation(train_size)
		train_loss = 0
        
		for i in range(train_size//batch_size) :

			optimizer.zero_grad()
			if isLSTM :
				framework.init_hidden(batch_size)
				
			batch_x = X_train[perm[i*batch_size:(i+1)*batch_size]]
			batch_y = y_train[perm[i*batch_size:(i+1)*batch_size]]
			
			pred_y = framework(batch_x)
			
			val_cc2 = concord_cc2(pred_y[:, 0], batch_y[:, 0])
			aro_cc2 = concord_cc2(pred_y[:, 1], batch_y[:, 1])
			
			loss = 1 - (val_cc2 + aro_cc2)/2
			loss.backward()
			optimizer.step()
			
			train_loss += loss

		train_loss /= (train_size//batch_size)
		train_loss = train_loss.detach().numpy()
		train_loss_list.append(train_loss)

		perm_test = np.random.permutation(test_size)
		test_loss = 0

		for i in range(test_size//batch_size) :

			if isLSTM :
				framework.init_hidden(batch_size)
			batch_x = X_test[perm_test[i*batch_size:(i+1)*batch_size]]
			batch_y = y_test[perm_test[i*batch_size:(i+1)*batch_size]]
			
			pred_y = framework(batch_x)
			
			val_cc2 = concord_cc2(pred_y[:, 0], batch_y[:, 0])
			aro_cc2 = concord_cc2(pred_y[:, 1], batch_y[:, 1])
			
			loss = 1 - (val_cc2 + aro_cc2)/2

			test_loss += loss

		test_loss /= (test_size//batch_size)
		test_loss = test_loss.detach().numpy()
		test_loss_list.append(test_loss)

		# check early stopping
		if epoch % period == 0:
			print('epoch:{} train loss:{} test loss:{}'.format(epoch, train_loss, test_loss))
			if(best_loss > test_loss) :
				torch.save(framework.state_dict(), "./Model_LSTM/model-ckpt-best.txt")
				best_loss = test_loss
			torch.save(framework.state_dict(), "./Model_LSTM/model-ckpt.txt")
else :

	print("Evalutaion Start ...............")
	batch_size = 1
	if isLSTM :
		framework.load_state_dict(torch.load("./Model_LSTM/model-ckpt-best.txt"))
		framework.eval()
		framework.init_hidden(batch_size)
	else :
		framework.load_state_dict(torch.load("./Model_Neural/model-ckpt-best.txt"))
		framework.eval()

	for param in framework.parameters():
		param.requires_grad = False
		
	predictions = []
	for i in range(test_size//batch_size) :
		
		if isLSTM :
			framework.init_hidden(batch_size)
		batch_x = X_test[i*batch_size:(i+1)*batch_size]
		batch_y = y_test[i*batch_size:(i+1)*batch_size]
		    
		pred_y = framework(batch_x)

		predictions.append(pred_y)
		
	predictions = torch.stack(predictions).squeeze(dim=1)
	plot_two(predictions[:, 0], y_test[:, 0], "Valence")
	plot_two(predictions[:, 1], y_test[:, 1], "Arousal")		
	valence_cc2 = concord_cc2(predictions[:, 0], y_test[:, 0])
	arousal_cc2 = concord_cc2(predictions[:, 1], y_test[:, 1])
	
	print('Concordance on valence : {}'.format(valence_cc2))
	print('Concordance on arousal : {}'.format(arousal_cc2))
	print('Concordance on total : {}'.format((arousal_cc2+valence_cc2)/2))
	
	loss_fn = torch.nn.MSELoss()
	
	mse_valence = loss_fn(predictions[:, 0], y_test[:, 0])
	mse_arousal = loss_fn(predictions[:, 1], y_test[:, 1])
	
	print('MSE Arousal : {}'.format(mse_arousal))
	print('Valence Arousal : {}'.format(mse_valence))        
