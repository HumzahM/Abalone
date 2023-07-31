import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import statistics

#matplotlib.interactive(True)

from torch import nn
import torch.optim as optim

import horovod.torch as hvd

hvd.init()

#https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
#https://archive.ics.uci.edu/ml/datasets/abalone

raw_data = np.genfromtxt("abalone.csv", delimiter=',', dtype=float, encoding=None)
#after spending time fighting with this over the raw data, I went into excel and used =IF(A1="M", 3, IF(A1="F", 2, 1))
ring_data = raw_data[1:, 8]
body_data = raw_data[1:,0:8]

body_train, body_test, ring_train, ring_test = train_test_split(body_data, ring_data, test_size=.05, random_state=26)

class Data(Dataset):
    def __init__(self, body, rings):
        self.X = torch.from_numpy(body.astype(np.float32))
        self.y = torch.from_numpy(rings.astype(np.float32))
        self.len = self.X.shape[0] 
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
    

batch_size = 64 #random number

#load the data 
train_data = Data(body_train, ring_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size)

test_data = Data(body_test, ring_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(8, 144)
        self.layer_2 = nn.Linear(144, 72)
        self.layer_3 = nn.Linear(72, 18)
        self.layer_4 = nn.Linear(18, 1)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        x = torch.nn.functional.relu(self.layer_4(x))

        return x

torch.cuda.set_device(hvd.local_rank())

learning_rate = 0.01
num_epochs = 100
fail = True
failCounter = 0

while(fail):
    loss_values = []
    counter = 0
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    for epoch in range(num_epochs):
        counter += 1
        if(counter % 10 == 0):
            print(counter)
        if(counter > (20) and loss_values[counter] > 50):
            failCounter += 1
            print("fail " + str(failCounter))
            print("Restarting")
            break
        for X, y in train_dataloader:
            # zero the parameter gradients
            #X = X.to(device)
            #y = y.to(device)
            optimizer.zero_grad()
        
            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
        if counter == num_epochs:
            fail = False

print("Training Complete")

#plot losses

step = np.linspace(0, 100, 62*num_epochs) #need to do more formula not hard coded

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("step wise loss.png")

import itertools
y_pred = []
accurate = 0
total = 0
error = []
abserror = []

model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X.to(device).float())
        y_pred.append(outputs[0])

    for num in range(len(y_pred)):
        total += 1
        abserror.append(abs(y_pred[num] - ring_test[num]).item())
        error.append((y_pred[num] - ring_test[num]).item())
        if(abs(y_pred[num] - ring_test[num]).item() < 0.5):
            accurate += 1
    print("Total accuracy is:")
    print(accurate / total)
    print("Mean ABS Error:")
    print(str(statistics.mean(abserror)))

    plt.figure()
    plt.hist(error)
    plt.text(-10, 50, "Total accuracy is: " + str(accurate/total)) 
    plt.text(-10, 40, "Mean ABS Error: "  + str(statistics.mean(abserror)))
    plt.title("Difference")
    plt.savefig("histogram.png")

for name, param in model.named_parameters():
    if param.requires_grad:
        print("name: " + str(name))
        print("param data: " + str(param.data))
    
