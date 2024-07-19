
import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from numpy import vstack
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.optim import SGD
from torch.nn import BCELoss

class PeopleDataset(T.utils.data.Dataset):
    def __init__(self, src_file, num_rows=None):
        df = pd.read_csv(src_file)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        print(df.columns)
        df.Class = df.Class.astype('float64')
        y_tmp = df['Class'].values
        x_tmp = df.drop('Class', axis=1).values


        self.x_data = T.tensor(x_tmp,dtype=T.float64).to(device)
        self.y_data = T.tensor(y_tmp,dtype=T.float64).to(device)

        print(type(self.x_data))
        print(len(self.x_data))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx].type(T.FloatTensor)
        pol = self.y_data[idx].type(T.LongTensor)
        sample = [preds, pol]
        return sample

class MLPUpdated(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

fed_acc, fed_pre, fed_recall, fed_f1 = list(), list(), list(), list()
reg_acc, reg_pre, reg_recall, reg_f1 = list(), list(), list(), list()
split_acc, split_pre, split_recall, split_f1 = list(), list(), list(), list()

mp = {}

"""# Federated Learning"""

# Can be changed to 12,16,24,32
num_clients = 32
# Change it to 3, 6, 10, 16
num_selected = 16
num_rounds = 50
epochs = 5
batch_size = 1024
device = "cpu"
device = torch.device(device)
fed_acc, fed_pre, fed_recall, fed_f1 = list(), list(), list(), list()

# Dividing the training data into num_clients, with each client having equal number of data
traindata = PeopleDataset('/content/drive/MyDrive/Credit Card Fraud Detection Dataset/creditcard_train.csv')
print(len(traindata))
traindata_split = torch.utils.data.random_split(traindata, [int(len(traindata) / num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]


test_file = '/content/drive/MyDrive/Credit Card Fraud Detection Dataset/creditcard_test.csv'
test_ds = PeopleDataset(test_file)
test_loader = T.utils.data.DataLoader(test_ds,batch_size=batch_size, shuffle=True)

def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            binary_loss = torch.nn.BCEWithLogitsLoss()
            target = target.unsqueeze(1)
            target = target.float()
            loss = binary_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    actuals, predictions = list(), list()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = global_model(data)
            binary_loss = torch.nn.BCEWithLogitsLoss()
            target = target.unsqueeze(1)
            target = target.float()
            test_loss += binary_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            actual = target.numpy()
            pr = output.detach().numpy()
            pr = pr.round()
            predictions.append(pr)
            actuals.append(actual)

    test_loss /= len(test_loader.dataset)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    # calculate precision
    prescision = precision_score(actuals, predictions)
    # calculate recall
    recall = recall_score(actuals, predictions)
    # calculate f1
    f1 = f1_score(actuals, predictions)
    fed_acc.append(acc)
    fed_pre.append(prescision)
    fed_recall.append(recall)
    fed_f1.append(f1)
    print()
    print(confusion_matrix(actuals, predictions))
    return test_loss, acc, prescision, recall, f1

###########################################
#### Initializing models and optimizer  ####
############################################

#### global model ##########
# global_model =  MLPUpdated().cuda()
global_model = MLPUpdated().to(device)

############## client models ##############
# client_models = [ MLPUpdated().cuda() for _ in range(num_selected)]
client_models = [ MLPUpdated().to(device) for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model

############### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.01) for model in client_models]

print(len(fed_acc))
print(len(fed_pre))
print(len(fed_recall))
print(len(fed_f1))

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
# Runnining FL

import time
start_time = time.time()
for r in range(num_rounds):
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epoch=epochs)

    losses_train.append(loss)
    # server aggregate
    server_aggregate(global_model, client_models)

    test_loss, acc, prescision, recall, f1= test(global_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f | test prescision: %0.3f | test recall: %0.3f | test f1: %0.3f' % (loss / num_selected, test_loss, acc, prescision, recall, f1))

print("--- %s seconds ---" % (time.time() - start_time))
# time[24]['fed'] = (time.time() - start_time)

print(len(fed_acc))
print(len(fed_pre))
print(len(fed_recall))
print(len(fed_f1))
# print(mp)
# print(len(mp))

mp[num_selected] = {'fed_acc': fed_acc, 'fed_pre': fed_pre, 'fed_recall': fed_recall, 'fed_f1': fed_f1}
print(mp)
print(len(mp))



"""# Split Federated Learning"""

reg_acc, reg_pre, reg_recall, reg_f1 = list(), list(), list(), list()

# train the model
def train_model(train_dl, model, epochs):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(epochs):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            targets = targets.unsqueeze(1)
            targets = targets.float()
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        acc, prescision, recall, f1 = evaluate_model(test_loader, model)
        reg_acc.append(acc)
        reg_pre.append(prescision)
        reg_recall.append(recall)
        reg_f1.append(f1)

# evaluate the model

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    prescision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    return acc, prescision, recall, f1

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

traindata = PeopleDataset('/content/drive/MyDrive/Credit Card Fraud Detection Dataset/creditcard_train.csv')
print(len(traindata))

train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
rounds = 50
# for i in range(1,rounds):
model = MLPUpdated()
# train the model
train_model(train_loader, model, rounds)
# evaluate the model
acc, prescision, recall, f1 = evaluate_model(test_loader, model)
print('Accuracy: %.3f' % acc)
print('Prescision: %.3f' % prescision)
print('Recall: %.3f' % recall)
print('F1-Score: %.3f' % f1)

print(len(reg_acc))
print(len(reg_pre))
print(len(reg_recall))
print(len(reg_f1))

mp[0] = {'reg_acc': reg_acc, 'reg_pre': reg_pre, 'reg_recall': reg_recall, 'reg_f1': reg_f1}
print(mp)
print(len(mp[0]['reg_acc']))

"""# Split Learning


"""

num_clients = 32
num_selected = 16
num_rounds = 50
epochs = 5
batch_size = 1024
device = "cpu"
device = torch.device(device)
split_acc, split_pre, split_recall, split_f1 = list(), list(), list(), list()

# Dividing the training data into num_clients, with each client having equal number of data
traindata = PeopleDataset('/content/drive/MyDrive/Credit Card Fraud Detection Dataset/creditcard_train.csv')
print(len(traindata))
traindata_split = torch.utils.data.random_split(traindata, [int(len(traindata) / num_clients) for _ in range(num_clients)])
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True) for x in traindata_split]

test_file = '/content/drive/MyDrive/Credit Card Fraud Detection Dataset/creditcard_test.csv'
test_ds = PeopleDataset(test_file)
test_loader = T.utils.data.DataLoader(test_ds,batch_size=batch_size, shuffle=True)

class SplitNN(torch.nn.Module):
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        self.outputs = [None]*len(self.models)
        self.inputs = [None]*len(self.models)
        super().__init__()

    def forward(self, x):
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])
        self.inputs[1] = self.outputs[0].detach().requires_grad_()
        self.inputs[1] = self.inputs[1].requires_grad_()
        self.outputs[1] = self.models[1](self.inputs[1])

        return self.outputs[-1]

    def backward(self):
        grad_in = self.inputs[1].grad
        self.outputs[0].backward(grad_in)


    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    @property
    def location(self):
        return self.models[0].location if self.models and len(self.models) else None

batch_size = 1024
device = "cpu"



torch.manual_seed(0)  # Define our model segments
input_size = 30
hidden_sizes = [32, 16, 8]
output_size = 1
models = [
    nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Dropout(0.1),
    ),
    nn.Sequential(
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], output_size),
                nn.Sigmoid()
    )
]
# Create optimisers for each segment and link to them
optimizers = [
    optim.SGD(model.parameters(), lr=0.01,)
    for model in models
]


# Server Model
server_model = models[1]
# print(server_model)


client_models = []
for i in range(num_selected):
    model = models[0]
    client_models.append(model)

print(len(client_models))

opt = [optim.SGD(model.parameters(), lr=0.01) for model in client_models]
server_opt = optim.SGD(server_model.parameters(), lr=0.01)

def client_update(splitNN, train_loader, epoch=5):
    """
    This function updates/trains client model on client data
    """
    splitNN.train()
    for e in range(epoch):
        print(e)
        loss_fin = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.shape[0], -1)
            #1) Zero our grads
            splitNN.zero_grads()

            #2) Make a prediction
            pred = splitNN.forward(data)

            #3) Figure out how much we missed by
            binary_loss = torch.nn.BCEWithLogitsLoss()
            target = target.unsqueeze(1)
            target = target.float()


            loss = binary_loss(pred, target)

            #4) Backprop the loss on the end layer
            loss.backward()

            #5) Feed Gradients backward through the nework
            splitNN.backward()

            #6) Change the weights
            splitNN.step()

            # return loss.item()
            loss_fin += loss
        server_dict = splitNN.models[1].state_dict()
        return loss_fin, server_dict

def test(splitNN, dataloader, dataset_name):
    splitNN.eval()
    correct = 0
    test_loss = 0
    predictions = list()
    actuals = list()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.view(data.shape[0], -1)
            output = splitNN.forward(data)
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            target = target.unsqueeze(1)
            target = target.float()
            actual = target.numpy()
            actuals.append(actual)
            pr = output.numpy()
            pr = pr.round()
            predictions.append(pr)
    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    prescision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    split_acc.append(acc)
    split_pre.append(prescision)
    split_recall.append(recall)
    split_f1.append(f1)



    print("{}: Accuracy {}/{} ({:.2f}%)".format(dataset_name,
                                                correct,
                                                len(dataloader.dataset),
                                                100. * correct / len(dataloader.dataset)))

print(len(split_acc))
print(len(split_pre))
print(len(split_recall))
print(len(split_f1))

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
# Runnining FL

num_rounds = 50
import time
start_time = time.time()
for r in range(num_rounds):
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    # client update
    loss = 0
    epochs = 5
    for i in tqdm(range(num_selected)):
        print("Client Number: ", i)
        splitNN = SplitNN([client_models[i], server_model], [opt[i],server_opt ])
        temp_loss, server_dict = client_update(splitNN, train_loader[client_idx[i]], epoch=epochs)
        loss += temp_loss
        server_model.load_state_dict(server_dict)

    losses_train.append(loss)

    # Test on the last client
    test(splitNN, test_loader, "Test Data")
    print('%d-th round' % r)
print("--- %s seconds ---" % (time.time() - start_time))

print(len(split_acc))
print(len(split_pre))
print(len(split_recall))
print(len(split_f1))

mp[num_selected]['split_acc'] = split_acc
mp[num_selected]['split_pre'] = split_pre
mp[num_selected]['split_recall'] = split_recall
mp[num_selected]['split_f1'] = splf1acc
print(mp)
print(len(mp))
