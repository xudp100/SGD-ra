import torch
from data import getDataLoader
from train import setAllSeed, trainModule


if __name__ == '__main__':

    ##############################################
    # set device and datasets
    ##############################################
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    dsetName = 'ijcnn1'
    featureSize = 22
    
    ##############################################
    # set the parameters
    ##############################################
    seed = 42
    alpha = 0.9
    regularize = 'l2'
    learning_rate = 0.1
    epochs = 10

    ##############################################
    # load data and train module
    ##############################################
    setAllSeed(seed)
    tr_loader, te_loader = getDataLoader(dsetName, featureSize)
    trainModule(dsetName, featureSize, alpha, epochs, learning_rate, 
                tr_loader, te_loader, regularize, device)
    