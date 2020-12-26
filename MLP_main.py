import torch.optim.lr_scheduler as lr_scheduler
import argparse
import numpy as np
import torch

from feature_extractor import FeatureExtractor
from utils import read_data, convert_sparse_to_torch
from MLP_model import MLP
from loss import Loss
from MLP_train_val import training_loop



def main(feature, BCELoss, MSELoss, ContrastiveLoss, ContrastiveMargin):
    # check validity of loss function
    if BCELoss == 0 and MSELoss == 0:
        print('you must choose at least one loss to train the neural network...')
        return

    # read and split the data
    xtrain, ytrain, xvalid, yvalid = read_data()

    # extract TfIdf and Count Features based on N-Grams
    extractor = FeatureExtractor(xtrain, ytrain, xvalid, yvalid)

    # It cannot fit in GPU Memory
    DEVICE = 'cpu'


    if feature == 'TfIdf':
        xtrain = extractor.xtrain_TfIdf
        xvalid = extractor.xvalid_TfIdf

    elif feature == 'NGrams':
        xtrain = extractor.xtrain_Count
        xvalid = extractor.xvalid_Count

    elif feature == 'TfIdf_NGrams':
        xtrain = extractor.xtrain_TfIdf_Count
        xvalid = extractor.xvalid_TfIdf_Count

    else:
        print('unsupported feature, select one of (TfIdf, NGrams, TfIdf_NGrams)')
        return
    
    xtrain = convert_sparse_to_torch(xtrain, DEVICE)
    xvalid = convert_sparse_to_torch(xvalid, DEVICE)
    ytrain = torch.tensor(ytrain)
    yvalid = torch.tensor(yvalid)

    # parameters
    RANDOM_SEED = 42
    LEARNING_RATE = 0.01
    N_EPOCHS = 40

    LEARNING_RATE_DECAY = 0.9
    LEARNING_RATE_DECAY_STEP_SIZE = 3

    # instantiate the model
    torch.manual_seed(RANDOM_SEED)
    dim = xtrain.size()[1]
    model = MLP(dim).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=LEARNING_RATE_DECAY_STEP_SIZE, gamma=LEARNING_RATE_DECAY)

    loss = Loss(DEVICE, BCELoss, MSELoss, ContrastiveLoss, ContrastiveMargin)

    print('start training...')
    # start training
    model, optimizer, train_losses, valid_losses = training_loop(
                                                                xtrain, ytrain, xvalid, yvalid,
                                                                model,
                                                                loss,
                                                                optimizer,
                                                                scheduler,
                                                                N_EPOCHS,
                                                                DEVICE)


if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-feature', '--feature', type=str, help='features that will be extracted before classification (TfIdf, NGrams, TfIdf_NGrams)', default = 'NGrams')
    argparser.add_argument('-BCE', '--BCELoss', type=int, help='1 if you want to include BCE Loss, 0 otherwise, default is 1', default = 1)
    argparser.add_argument('-MSE', '--MSELoss', type=int, help='1 if you want to include MSE Loss, 0 otherwise, default is 0', default = 0)
    argparser.add_argument('-Contrastive', '--ContrastiveLoss', type=int, help='1 if you want to include Euclidean Contrastive Loss, 0 otherwise, default is 0', default = 0)
    argparser.add_argument('-ContrastiveMargin', '--ContrastiveMargin', type=int, help='margin used in contrastive loss, default is 1', default = 1)
    args = argparser.parse_args()

    main(args.feature, args.BCELoss, args.MSELoss, args.ContrastiveLoss, args.ContrastiveMargin)
        