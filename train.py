import os
import random
import torch
import datetime
import copy
import collections
import math
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


def setAllSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    np.random.seed(seed)                # Numpy module.
    random.seed(seed)                   # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    

class Logistic(nn.Module):
    def __init__(self, in_channels=784, out_channels=10):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


class Logisticmodule:
    def __init__(self, featureSize, device,
                  learning_rate,
                 ):
        self.module = Logistic(featureSize, 1)
        self.moduleInitial(self.module, device, learning_rate)
        
    def moduleInitial(self, module, device, learning_rate):
        module.loss_func = torch.nn.BCELoss()
        module.optimizer = torch.optim.SGD(module.parameters(),
                                           lr=learning_rate)
        module.metricFunc = Logisticmodule.metricFunc
        module.metric_name = 'acc'
        module.to(device)
        module.train()

    def metricFunc(label, pred):
        total = label.size(0)
        acc = (pred.ge(.5).float() == label).sum()
        return acc / total
    
    
def L1_Regular(params):
    res = 0
    for param in params:
        res += torch.norm(param)
    return res


def L2_Regular(params):
    res = 0
    for param in params:
        res += torch.norm(param, 1)
    return res


def averageStorageFunc(k, alpha, T):
    maxIdx = (1 - alpha) * T
    if k <= maxIdx:
        return 0
    else:
        return k - int(maxIdx)
    

def drawStoragePic(alpha, lengthList):
    T = len(lengthList)
    steps = range(T)  
    averageStorage = [averageStorageFunc(k+1, alpha, T) for k in steps]
    plt.figure()
    plt.plot(steps, lengthList,label=r'SGD$-r\alpha$', color='r')
    plt.plot(steps, averageStorage,label=r'SGD$-\alpha$', color='b')
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Length of the queue")
    plt.title(r'$\alpha$=%s, T=%d' % (alpha, T))


def validEpoch(module, dlValid, device, regularize, lamda):
    module.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    for features, labels in dlValid:     
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            predictions = module(features)
            loss = module.loss_func(predictions, labels)
            metric = module.metricFunc(labels, predictions)
            val_loss_sum += loss.detach().item()
            val_metric_sum += metric.item()
    
    return val_loss_sum / len(dlValid), val_metric_sum / len(dlValid)

    
def updateWeight(condition, ceilAlphaK, cur_w, al_w, pre_w):
    if condition:
        res = al_w - pre_w + cur_w
        if ceilAlphaK > 1:
            res = al_w + (cur_w  - pre_w) / ceilAlphaK  
        return res
    else:
        return al_w + (cur_w - al_w) / (ceilAlphaK + 1)
    
    
def limitFunc(alpha, T):
    return T - alpha * T


def trainModule(dsetName, featureSize, alpha, epochs, learning_rate, 
                train_loader, valid_loader, regularize, device, lamda=0.001
                ):
    module = Logisticmodule(featureSize, device, learning_rate).module
    validmodule = Logisticmodule(featureSize, device, learning_rate).module
    T = epochs * len(train_loader)
    alpha = alpha if alpha > 0 else 1 / (T+1)
    limitStep = limitFunc(alpha, T)
    
    cur_weight = None
    alpha_weight = copy.deepcopy(module.state_dict()) 
    weights_list = [alpha_weight]
    lengthList = list()
    print(r'Start Training, alpha=' + str(alpha))
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 32 + "%s" % nowtime)
    for epoch in range(epochs+1):
        # show the history every epoch
        validmodule.load_state_dict(alpha_weight)
        tr_loss, tr_metric = validEpoch(validmodule, train_loader, device, regularize, lamda)
        te_loss, te_metric = validEpoch(validmodule, valid_loader, device, regularize, lamda)
        print("[epoch = %d] Q_len: %.0f, loss: %.5f, acc: %.3f, test_loss: %.5f, test_acc: %.3f" % 
              (int(epoch), len(weights_list), tr_loss, tr_metric, te_loss, te_metric))
        
        if epoch != epochs:
            for step, (features, labels) in enumerate(train_loader, start=1):
                features, labels = features.to(device), labels.to(device)
                totalStep = epoch * len(train_loader) + step
                if totalStep==epochs* len(train_loader):
                    break
                module.optimizer.zero_grad()
                predictions = module(features)
                loss = module.loss_func(predictions, labels)    
                if regularize == 'l1':
                    loss += lamda * L1_Regular(module.parameters())
                elif regularize == 'l2':
                    loss += lamda * L2_Regular(module.parameters())
                loss.backward()
                module.optimizer.step()                   
                # update alpha_weight
                cur_weight = copy.deepcopy(module.state_dict())
                if totalStep + 1 <= limitStep:            
                    weights_list.append(cur_weight)
                pre_weight = collections.defaultdict(int)
                ceilAlphaK = math.ceil(alpha * totalStep)
                condition = math.ceil(alpha * (totalStep + 1)) == ceilAlphaK
                if condition:
                    pre_weight = weights_list.pop(0)
                for d in alpha_weight:
                    alpha_weight[d] = updateWeight(condition, ceilAlphaK,
                                                   cur_weight[d], alpha_weight[d],
                                                   pre_weight[d])
                lengthList.append(len(weights_list))
                
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=" * 32 + "%s" % nowtime)
    print('Finished Training...')
    drawStoragePic(alpha, lengthList)
    return
