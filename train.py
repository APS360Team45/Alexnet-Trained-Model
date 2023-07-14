import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import cv2


import torchvision.models

alexnet = torchvision.models.alexnet(pretrained=True)



    
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
e
    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path


def evaluate(net, loader, criterion): # this function is for evaluating a model based on a given dataset and criterion
    '''
    net --> model
    loader --> type: DataLoader with specified batches
    criterion --> loss function
    '''
    total_loss = 0.0
    total_accuracy = 0.0
    total_iter = 0.0
    for i, data in enumerate(loader, 0):
      inputs, labels = data
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      total_loss += loss.item()

      prediction = torch.round(outputs) # round predictions to 0, 1, 2, or 3

      corr = torch.eq(prediction, labels).sum() # sum all the matching indeces together, obtaining a tensor containing boolean values, then summing them together
      total_accuracy += int(corr) # add number of correct predictions to total accuracy
      total_iter += len(labels) # update iteration by adding batch_size (number of labels)

    accuracy = float(total_accuracy)/total_iter # obtain accuracy by dividing total number of correct predictions by total number of predictions
    loss = float(total_loss) / (i + 1) # obtain loss by dividing total CE loss per batch by number of iterations
    return loss, accuracy

def train(model, train_dataset, val_dataset, batch_size=64, lr=0.001, num_epochs=10, print_stat=False, use_cuda=False):
    torch.manual_seed(1000)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    train_loss = np.zeros(num_epochs)
    train_accuracy = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_accuracy = np.zeros(num_epochs)

    start_time = time.time()

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):

        running_loss = 0.0
        total_accuracy = 0.0
        total_iter = 0.0

        for i, data in enumerate(train_loader):
            imgs, labels = data
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            
            # convert the labels to floating point type
            labels = labels.float()

            out = model(imgs)
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            running_loss += loss.item()
            prediction = torch.round(out)# round predictions to 0, 1, 2, or 3
            corr = torch.eq(prediction, labels).sum() # sum all the matching indeces together, obtaining a tensor containing boolean values, then summing them together
            total_accuracy += int(corr)
            total_iter += len(labels)
        
        # save the current training information
        train_loss[epoch] = float(running_loss) / (i + 1)
        train_accuracy[epoch] = float(total_accuracy) / total_iter
        val_loss[epoch], val_accuracy[epoch] = evaluate(model, val_loader, criterion)

        print((f"Epoch {epoch + 1}: Train accuracy: {train_accuracy[epoch] * 100:.1f}%, Train loss: {train_loss[epoch]:.4f} | "+
          f"Validation accuracy: {val_accuracy[epoch] * 100:.1f}%, Validation loss: {val_loss[epoch]:.4f}"))
        
        n += 1
        model_path = get_model_name(model.name, batch_size, lr, n)
        #     torch.save(model.state_dict(), model_path)

    end_time = time.time()
    torch.save(model.state_dict(), model_path)
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

    # Save array of training/validation loss/accuracy
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

    np.savetxt("{}_train_accuracy.csv".format(model_path), train_accuracy)
    np.savetxt("{}_val_accuracy.csv".format(model_path), val_accuracy)


test_model_0 = FruitRipenessDetector()

train_dataset = torch.load('train_dataset.pth')
val_dataset = torch.load('val_dataset.pth')
test_dataset = torch.load('test_dataset.pth')


train_data = []

for image, label in train_dataset:
  # normalize pixel intensity values back to [0, 1]
  image = image / 2 + 0.5
  #convert from hsv to rgb
  image = cv2.cvtColor(image.numpy().transpose(1,2,0), cv2.COLOR_HSV2RGB)
  image = cv2.resize(image, (224, 224))
  image = torch.from_numpy(image.transpose(2,0,1))
  features = alexnet.features(image)
  train_data.append([features, label])

torch.save(train_data, "alex_features_train")

val_data = []

for image, label in val_dataset:
  # normalize pixel intensity values back to [0, 1]
  image = image / 2 + 0.5
  #convert from hsv to rgb
  image = cv2.cvtColor(image.numpy().transpose(1,2,0), cv2.COLOR_HSV2RGB)
  image = cv2.resize(image, (224, 224))
  image = torch.from_numpy(image.transpose(2,0,1))
  features = alexnet.features(image)
  val_data.append([features, label])

torch.save(val_data, "alex_features_val")

alex_data_train = torch.load("alex_features_train")
alex_data_val = torch.load("alex_features_val")


class Al_Net(nn.Module):
    def __init__(self, name):
        super(Al_Net, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(512*4*4, 328)
        self.fc2 = nn.Linear(328, 96)
        self.fc3 = nn.Linear(96, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x
    

train_dataset = torch.load('train_dataset.pth')
val_dataset = torch.load('val_dataset.pth')
test_dataset = torch.load('test_dataset.pth')


train_data = []

for image, label in train_dataset:
  # normalize pixel intensity values back to [0, 1]
  image = image / 2 + 0.5
  #convert from hsv to rgb
  image = cv2.cvtColor(image.numpy().transpose(1,2,0), cv2.COLOR_HSV2RGB)
  image = cv2.resize(image, (224, 224))
  image = torch.from_numpy(image.transpose(2,0,1))
  features = alexnet.features(image)
  train_data.append([features, label])

torch.save(train_data, "alex_features_train")

val_data = []

for image, label in val_dataset:
  # normalize pixel intensity values back to [0, 1]
  image = image / 2 + 0.5
  #convert from hsv to rgb
  image = cv2.cvtColor(image.numpy().transpose(1,2,0), cv2.COLOR_HSV2RGB)
  image = cv2.resize(image, (224, 224))
  image = torch.from_numpy(image.transpose(2,0,1))
  features = alexnet.features(image)
  val_data.append([features, label])

torch.save(val_data, "alex_features_val")

alex_data_train = torch.load("alex_features_train")
alex_data_val = torch.load("alex_features_val")


class Al_Net(nn.Module):
    def __init__(self, name):
        super(Al_Net, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(256, 512, 3)
        self.fc1 = nn.Linear(512*4*4, 328)
        self.fc2 = nn.Linear(328, 96)
        self.fc3 = nn.Linear(96, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)  # Flatten to [batch_size]
        return x
    
        
for i, data in enumerate(alex_data_train):
  feature, label = data
  alex_data_train[i] = (torch.from_numpy(feature.detach().numpy()), label)

for i, data in enumerate(alex_data_val):
  feature, label = data
  alex_data_val[i] = (torch.from_numpy(feature.detach().numpy()), label)

al_net = Al_Net("al_net")

train(al_net, alex_data_train, alex_data_val, batch_size=64, num_epochs=30, lr=0.001)