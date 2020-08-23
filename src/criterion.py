import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, loss_type="ce"):
        super().__init__()

        self.loss_type = loss_type
        if loss_type == "ce":
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == "bce":
            self.loss = nn.BCELoss()
        elif loss_type == "bcewl":
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        if self.loss_type == "ce":
            input_ = input["multiclass_proba"]
            target = target.argmax(1).long()
        elif self.loss_type == "bce":
            input_ = input["multilabel_proba"]
            target = target.float()
        elif self.loss_type == "bcewl":
            input_ = input["multilabel_proba"]
            target = target.float()

        return self.loss(input_, target)


# class Cutmix_Loss(nn.Module):
#     def __init__(self, loss_type="ce"):
#         super().__init__()

#         self.loss_type = loss_type
#         if loss_type == "ce":
#             self.loss = nn.CrossEntropyLoss()
#         elif loss_type == "bce":
#             self.loss = nn.BCELoss()
#         elif loss_type == "bcewl":
#             self.loss = nn.BCEWithLogitsLoss()

#     def forward(self, input, targets):
#         target1 = targets[0]["multiclass_proba"]
#         target2 = targets[1]["multiclass_proba"]
#         lam = targets[2]["multiclass_proba"]

#         if self.loss_type == "ce":
#             target1 = target1.argmax(1).long()
#             target2 = target2.argmax(1).long()
#         else:
#             target1 = target1.float()
#             target2 = target2.float()

#         return lam * self.loss(input, target1) + lam * self.loss(input, target2)


# class Mixup_Loss(nn.Module):
#     def __init__(self, loss_type="ce"):
#         super().__init__()

#         self.loss_type = loss_type
#         if loss_type == "ce":
#             self.loss = nn.CrossEntropyLoss()
#         elif loss_type == "bce":
#             self.loss = nn.BCELoss()
#         elif loss_type == "bcewl":
#             self.loss = nn.BCEWithLogitsLoss()

#     def forward(self, input, targets):
#         target1 = targets[0]["multiclass_proba"]
#         target2 = targets[1]["multiclass_proba"]
#         lam = targets[2]["multiclass_proba"]

#         if self.loss_type == "ce":
#             target1 = target1.argmax(1).long()
#             target2 = target2.argmax(1).long()
#         else:
#             target1 = target1.float()
#             target2 = target2.float()

#         return lam * self.loss(input, target1) + (1-lam) * self.loss(input, target2)


class PANNsLoss(nn.Module):
    def __init__(self, loss_type="ce"):
        super().__init__()
          
        if loss_type == "ce":
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == "bce":
            self.loss = nn.BCELoss()
        elif loss_type == "bcewl":
            self.loss = nn.BCEWithLogitsLoss()


    def forward(self, input, target):

        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        return self.loss(input_, target)