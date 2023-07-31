import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import statistics as stats 
from torch import nn
import torch.optim as optim


#https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
#https://archive.ics.uci.edu/ml/datasets/abalone

raw_data = np.genfromtxt("abalone.csv", delimiter=',', dtype=float, encoding=None)
#after spending time fighting with this over the raw data, I went into excel and used =IF(A1="M", 3, IF(A1="F", 2, 1))
ring_data = raw_data[1:, (0,8)]
body_data = raw_data[1:,1:8]

#print(body_data)

body_train, body_test, ring_train, ring_test = train_test_split(body_data, ring_data, test_size=.025, random_state=26)

class Data(Dataset):
    def __init__(self, body, rings):
        self.y = torch.from_numpy(body.astype(np.float32))
        self.X = torch.from_numpy(rings.astype(np.float32))
        self.len = self.X.shape[0] 
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
    
batch_size = 128 #random number

#load the data 
train_data = Data(body_train, ring_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)

test_data = Data(body_test, ring_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(2, 72)
        self.layer_2 = nn.Linear(72, 28)
        self.layer_3 = nn.Linear(28, 7)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        return x
    
if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")
#0.0001 works best in isolation
num_epochs = 50
fail = True
failCounter = 0

while(fail):
    learning_rate = 0.0001
    loss_values = []
    counter = 0
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    loss_fn = nn.L1Loss()
    for epoch in range(num_epochs):
        counter += 1
        if(counter % 5 == 0):
            print(loss_values[counter])
            print(counter)
        if(counter > (5) and loss_values[counter] > 0.3):
            failCounter += 1
            print("fail " + str(failCounter))
            print("Restarting")
            break
        if(epoch == 20):
            learning_rate = learning_rate/10
        for X, y in train_dataloader:
            # zero the parameter gradients
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
 
            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            loss_values.append(loss.item())
            optimizer.step()
        if counter == num_epochs:
            fail = False

print("Training Complete")

import itertools
y_pred = []
accurate = 0
total = 0
x_test = []
import sklearn.linear_model as LR
actuals = np.zeros((7, 105))
preds = np.zeros((7, 105))

model.eval()
counter = 0
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X.to(device).float())
        y_pred.append(outputs[0])

    for num in range(len(y_pred)):
        counter += 1
        print(ring_test[num])
        print(y_pred[num])
        """
        for i in range(7):
            actuals[i-1][counter] = (body_test[num][i-1])
            preds[i-1][counter] = (y_pred[num][i-1])
        """
        print(body_test[num])
        print("------")
        counter += 1

plt.plot(loss_values)
plt.show()

for i in range(7):
    print("R^2 Val: " + str(i))
    model = LR.LinearRegression.fit(actuals[i-1], preds[i-1])
    print(model.score(actuals[i-1], preds[i-1]))
    