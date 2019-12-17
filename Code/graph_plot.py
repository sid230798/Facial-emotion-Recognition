import numpy as np
import matplotlib.pyplot as plt 


def read_file(file_path):
	with open(file_path) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 
	content = content[4:]
	output = []
	for line in content:
		line = line.split("\t")
		reading_id 	= 	int(line[0])
		mote_id 	= 	int(line[1])
		humidity 	= 	float(line[2])
		temp 		= 	float(line[3])
		label 		= 	int(line[4])
		output.append([reading_id,mote_id,humidity,temp,label])
	output = np.asarray(output)
	return output
	
def get_XY(matrix):
	X = matrix[:,[2,3]]
	Y = matrix[:,[4]].T[0]
	Y = Y.astype(int)
	return X,Y

def get_normalized_X(X):
	return (X - np.amin(X, axis=0))/np.amax(X, axis=0)

def get_standardized_X(X):
	return (X - np.mean(X, axis=0))/np.std(X, axis=0)

# create data of the "look_back" length from time-series, "ts"
# and the next "pred_length" values as labels
def create_subseq(ts, look_back, pred_length):
    sub_seq, next_values = [], []
    for i in range(len(ts)-look_back-pred_length):  
        sub_seq.append(ts[i:i+look_back])
        next_values.append(ts[i+look_back:i+look_back+pred_length])
    return np.array(sub_seq), np.array(next_values)

def get_gradient_form_data(X,Y):
	X_tmp = X[1:] - X[:-1] 
	return X_tmp,Y[1:]

def visualize(X, Y):

	plt.style.use('ggplot')
	plt.figure(figsize=(15,5))
	plt.xlabel('time')
	plt.ylabel('ECG\'s value')
	plt.plot(np.arange(len(X)), X, color='b')
	plt.ylim(-3, 3)
	x = np.where(Y == 1)[0]
	y1 = [-3]*len(x)
	y2 = [3]*len(x)
	plt.fill_between(x, y1, y2, facecolor='g', alpha=.3)
	plt.show()

def visualize_duel(X1, X2, Y):
	fig, axes = plt.subplots(nrows=2, figsize=(15,10))

	axes[0].plot(X1,color='b',label='original data')
	axes[0].set_xlabel('time')
	axes[0].set_ylabel('Humidites\'s value')
	axes[0].set_ylim(-3, 3)
	x = np.where(Y == 1)[0]
	y1 = [-3]*len(x)
	y2 = [3]*len(x)
	axes[0].fill_between(x, y1, y2, facecolor='g', alpha=.3)

	axes[1].plot(X2, color='r',label='Mahalanobis Distance')
	axes[1].set_xlabel('time')
	axes[1].set_ylabel('Mahalanobis Distance')
	axes[1].set_ylim(0, 1000)
	y1 = [0]*len(x)
	y2 = [1000]*len(x)
	axes[1].fill_between(x, y1, y2, facecolor='g', alpha=.3)

	plt.legend(fontsize=15)
	plt.show()

def visualize_with_readings(temp,humidity,label,readings = None):
	try:
		if readings==None:
			readings = list(range(1,len(humidity)+1))
	except:
		pass
	print(len(readings),len(temp),len(humidity),len(label))
	positive_readings = readings*(1-label)
	negative_humidity = readings*label

	negative_humidity = humidity*label
	positive_humidity = humidity*(1-label)
	positive_temp = temp*(1-label)
	negative_temp = temp*label
	plt.plot(readings, positive_humidity, label = "Valid data") 
	plt.plot(readings, negative_humidity, label = "Invalid data") 
	plt.legend() 
	plt.show()
	plt.clf()
	plt.plot(readings, positive_temp, label = "Valid data") 
	plt.plot(readings, negative_temp, label = "Invalid data") 
	plt.legend() 
	plt.show()


def visualize_graph(input_data):
	plt.clf()
	readings =  input_data[:,[0]].T[0]
	temp     =  input_data[:,[3]].T[0]
	humidity =  input_data[:,[2]].T[0]
	label    =  input_data[:,[4]].T[0]
	visualize_with_readings(temp,humidity,label,readings)

def plot_graph(X, Y1, Y2):
	plt.clf()
	plt.plot(X, Y1, label = "Train Loss")
	plt.plot(X, Y2, label = "Test Loss")

	plt.xlabel('Epochs')
	plt.ylabel('Loss')

	plt.ylim(0.013, 0.1)
	plt.title("LSTM Network Training")
	plt.legend()
	plt.show()   

def get_error(model,X,Y):
	predicted_Y = model.predict(X)
	error = Y-predicted_Y
	erro = error.tolist()
	total_err = np.sum(np.abs((error)))
	percentage = float(total_err)/len(Y)
	return total_err,percentage

if __name__ == '__main__' :

	file_path = "LSTM_loss.txt"
	f = open(file_path, 'r')

	X = []
	X1 = []
	X2 = []
	for line in f:
		line = line.split()
		X.append(int(line[0]))
		X1.append(float(line[1]))
		X2.append(float(line[2]))

	#print(min(X1), max(X1), min(X2), max(X2))
	plot_graph(X, X1, X2)
