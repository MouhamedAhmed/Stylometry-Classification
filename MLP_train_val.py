import numpy as np
from datetime import datetime 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import get_acc

###############
# train iteration
def train(xtrain, ytrain, model, loss, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    xtrain = xtrain.to(device)
    
    # forward pass
    optimizer.zero_grad()
    yhat = model(xtrain)

    # cost
    total_loss = loss(yhat, ytrain)

    # train accuracy
    accuracy = get_acc(ytrain, yhat)
    
    # backward pass
    total_loss.backward()
    optimizer.step()
    
    return model, optimizer, total_loss.item(), accuracy
    
# validate iteration
def validate(xvalid, yvalid, model, loss, optimizer, device):
    '''
    Function for the validation step of the training loop
    '''
    model.eval()
    xvalid = xvalid.to(device)
    yvalid = yvalid.to(device)
    
    # forward pass
    yhat = model(xvalid)

    # cost
    total_loss = loss(yhat, yvalid)

    # valid accuracy
    accuracy = get_acc(yvalid, yhat)
        
    return model, total_loss.item(), accuracy

def training_loop(xtrain, ytrain, xvalid, yvalid, model, loss, optimizer, scheduler, epochs, device):
    '''
    Function defining the entire training loop
    '''
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    
    # Train model
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss , train_acc = train(xtrain, ytrain, model, loss, optimizer, device)
        train_losses.append(train_loss)
        # validation
        with torch.no_grad():
            model, valid_loss, valid_acc = validate(xvalid, yvalid, model, loss, optimizer, device)
            valid_losses.append(valid_loss)

        print(
              f'{datetime.now().time().replace(microsecond=0)} --- '
              f'Epoch: {epoch} '
              f'Train loss: {train_loss:.4f} '
              f'Valid loss: {valid_loss:.4f} '
              f'Train acc: {train_acc:.4f} '
              f'Valid acc: {valid_acc:.4f} '
              )
              
        torch.save(model, 'model')
        scheduler.step()
    
    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, train_losses, valid_losses

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')
    plt.show()