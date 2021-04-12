# Copyright 2021 Justin Barry

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Please send emails to this account with questions / comments.

import pandas as pd
import model

import torch.utils.data
import torch.nn as nn
import torch.autograd
import torch.optim as optim

data_path = '../PredictiveModel/dataset/data.csv'  # Set to your dataset CSV

selections = pd.read_csv(data_path)
print(data_path)

selections.head()

selections.plot(x='quarter_of_day', y='selected_object')

dummy_fields = ['directing_page', 'selected_object']  # set one hot values for these fields

for each in dummy_fields:
    dummies = pd.get_dummies(selections[each], prefix=each, drop_first=False)
    selections = pd.concat([selections, dummies], axis=1)

fields_to_drop = ['directing_page', 'selected_object']  # drop fields that have been converted to one hot
data = selections.drop(fields_to_drop, axis=1)

print(data.head())

quant_features = ['quarter_of_day']  # define quantity features and normalize
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

data.head()
print(data.describe())
# Set test data set and training data
# Here the test data is the last 100 values
test_data = data[-100:]

# Now remove the test data from the data set
data = data[:-100]

# Separate the data into features and targets
target_fields = ['selected_object_4', 'selected_object_5', 'selected_object_6', 'selected_object_7']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

print(features.shape);
print(test_features.shape)

train_targets = targets.to_numpy()
train_features = features.to_numpy()
val_features = test_features.to_numpy()
val_targets = test_targets.to_numpy()

nnmodel = model.ANN()

print(nnmodel)

x_train = torch.from_numpy(train_features)
y_train = torch.from_numpy(train_targets)

x_test = torch.from_numpy(val_features)
y_test = torch.from_numpy(val_targets)

train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(nnmodel.parameters(), lr=0.000001)

# lines 1 to 6
epochs = 20000
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# lines 7 onwards
nnmodel.train()  # prepare model for training
for epoch in range(epochs):
    train_loss = 0.0

    correct = 0
    total = 0
    for data, target in train_loader:
        data = torch.autograd.Variable(data).float()
        target = torch.autograd.Variable(target).type(torch.FloatTensor)
        output = nnmodel(data)
        predicted = (torch.round(output.data[0]))
        total += len(target)
        correct = torch.sum(predicted == target)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    accuracy = 100 * correct / float(total)
    train_acc_list.append(accuracy)
    train_loss_list.append(train_loss)
    print('\rEpoch: {} \tTraining Loss: {:.4f}\t Acc: {:.2f}%'.format(epoch + 1, train_loss, accuracy), end="")
    epoch_list.append(epoch + 1)

correct = 0
total = 0
valloss = 0
nnmodel.eval()

with torch.no_grad():
    for data, target in test_loader:
        data = torch.autograd.Variable(data).float()
        target = torch.autograd.Variable(target).type(torch.FloatTensor)

        output = nnmodel(data)
        loss = loss_fn(output, target)
        valloss += loss.item() * data.size(0)

        predicted = (torch.round(output.data[0]))
        total += len(target)
        correct = torch.sum(predicted == target)

    valloss = valloss / len(test_loader.dataset)
    accuracy = 100 * correct / float(total)
    print(accuracy)
