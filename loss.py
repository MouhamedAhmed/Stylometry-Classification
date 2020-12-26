import random
import torch
import torch.nn as nn

class Loss(torch.nn.Module):
    def __init__(self, device, BCELoss_select, MSELoss_select, contrasiveLoss_select, contrastive_margin=1.0):
        super(Loss, self).__init__()
        self.margin = contrastive_margin
        self.device = device
        self.BCE_loss_criterion = nn.BCELoss()
        self.MSE_loss_criterion = nn.MSELoss()
        self.BCELoss_select = BCELoss_select
        self.MSELoss_select = MSELoss_select
        self.contrasiveLoss_select = contrasiveLoss_select
        self.euclidean_contrastive_loss_criterion = EucContrastiveLoss(device = device, n_classes = 2, margin = contrastive_margin)

        
    def forward(self, y_hat, y_true):
        y_true = y_true.to(self.device)
        # convert one-hot vector to matrix
        y_true_mat = torch.zeros(y_hat.size()[0], y_hat.size()[1])
        y_true_mat[range(y_true_mat.shape[0]), y_true] = 1
        y_true_mat = y_true_mat.to(self.device)
        
        BCE_loss = self.BCE_loss_criterion(y_hat, y_true_mat)
        MSE_loss = self.MSE_loss_criterion(y_hat, y_true_mat)
        euc_contrastive_loss = self.euclidean_contrastive_loss_criterion(y_hat, y_true)
        
        total_loss = BCE_loss*self.BCELoss_select + MSE_loss*self.MSELoss_select + euc_contrastive_loss*self.contrasiveLoss_select
        
        return total_loss


class EucContrastiveLoss(torch.nn.Module):
    def __init__(self, device, n_classes, margin=1.0):
        super(EucContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.n_classes = n_classes
        self.pairwise_distance = nn.PairwiseDistance(keepdim=True)

    def forward(self, y_hat, y_true):
        classes = set(y_true.cpu().detach().numpy())
        if len(list(classes)) > 1:
                
            same_class_chunk_1 = torch.empty(1,self.n_classes).to(self.device)
            same_class_chunk_2 = torch.empty(1,self.n_classes).to(self.device)

            for c in classes:
                class_hats = y_hat[y_true == c]
                class_split = torch.chunk(class_hats, 2, dim=0)
                if len(class_split) > 1:
                    class_split_2 = class_split[1]
                    class_split_1 = class_split[0][0:class_split_2.size()[0]]
                else:
                    class_split_1 = class_split[0]
                    class_split_2 = class_split[0]
                
                for s1 in class_split_1:
                    same_class_chunk_1 = torch.cat((same_class_chunk_1, s1.unsqueeze(0)), dim=0)
                for s2 in class_split_2:
                    same_class_chunk_2 = torch.cat((same_class_chunk_2, s2.unsqueeze(0)), dim=0)
            
            same_class_chunk_1 = same_class_chunk_1[1:]
            same_class_chunk_2 = same_class_chunk_2[1:]
            
            diff_class_chunk_1 = torch.empty(1,self.n_classes).to(self.device)
            diff_class_chunk_2 = torch.empty(1,self.n_classes).to(self.device)
            for _ in range(same_class_chunk_1.size()[0]):
                # pick two different classes randomly
                class_1, class_2 = random.sample(classes, 2)
                class_1_samples = y_hat[y_true == class_1]
                class_2_samples = y_hat[y_true == class_2]
                
                # pick 2 random elements from the 2 different classes
                class_1_choice = random.choice(class_1_samples)
                class_2_choice = random.choice(class_2_samples)

                # append the examples to the lists

                diff_class_chunk_1 = torch.cat((diff_class_chunk_1, class_1_choice.unsqueeze(0)), dim=0)
                diff_class_chunk_2 = torch.cat((diff_class_chunk_2, class_2_choice.unsqueeze(0)), dim=0)                

                
            diff_class_chunk_1 = diff_class_chunk_1[1:]
            diff_class_chunk_2 = diff_class_chunk_2[1:]

            # create tensors for losses
            same_label = torch.zeros(same_class_chunk_1.size()[0], 1)
            diff_label = torch.ones(diff_class_chunk_1.size()[0], 1)
            
            label = torch.cat((same_label, diff_label), 0).to(self.device)
            chunk1 = torch.cat((same_class_chunk_1, diff_class_chunk_1), 0)
            chunk2 = torch.cat((same_class_chunk_2, diff_class_chunk_2), 0)
            
            

            # euc loss calculation
            euclidean_distance = self.pairwise_distance(chunk1, chunk2)
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            loss_contrastive = loss_contrastive.to(self.device)
            return loss_contrastive
            

        return torch.FloatTensor([0])[0].to(self.device)

        
        
        
    