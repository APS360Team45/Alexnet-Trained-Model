import matplotlib.pyplot as plt
import numpy as np

def plot(path): # this function is for plotting the training and validation losses and accuracies during training

    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    train_accuracy = np.loadtxt("{}_train_accuracy.csv".format(path))
    val_accuracy = np.loadtxt("{}_val_accuracy.csv".format(path))

    n = len(train_loss) # number of epochs

    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Train vs Validation Accuracy")
    plt.plot(range(1,n+1), train_accuracy, label="Train")
    plt.plot(range(1,n+1), val_accuracy, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

plot("model_al_net_bs64_lr0.001_epoch30")