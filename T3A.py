import torch
import torch.nn as nn
import torch.nn.functional as F


class T3A():
    def __init__(self, last_layer: nn.Linear, k=0, num_classes=60):
        '''

        :param last_layer:
        :param k: select k max when get features. if k <=0, select all.
        '''
        self.last_layer = last_layer
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.supports = {}
        self.supports_entropy = {}
        self.num_classes = num_classes

        for i in range(num_classes):
            self.supports[i] = []
            self.supports_entropy[i] = []

    def forward(self, x, adapt=False, use_T3A=False):
        '''

        :param x:
        :param adapt: whether use this batch to update parameter of this model
        :return:
        '''
        if adapt:
            self.update_supports(x)
        if not use_T3A:
            return self.last_layer(x)
        weight = self.get_supports().permute(1, 0)  # D, num_classes
        bias = self.last_layer.bias.data

        # X N,D
        return x @ weight + bias

    def get_supports(self, k=None):
        if k is None:
            k = self.k
        weight = []
        for i in range(self.num_classes):
            all = torch.cat(self.supports[i], dim=0).to(self.device)  # N, D

            if k <= 0:
                center = all.mean(0)
            else:
                entropy = torch.tensor(self.supports_entropy[i], device=self.device)  # N
                _, indices = torch.sort(entropy)
                all = all[indices, :]
                all = all[torch.arange(0, k), :]
                center = all.mean(0)

                # clean the supports
                self.supports[i] = []
                self.supports_entropy[i] = []
                self.update_supports(all)

            weight.append(center)

        # weight num_classes, D
        return weight

    def update_supports(self, x):
        '''

        :param x: N, D
        :return:
        '''
        pre = self.last_layer(x)  # N, D
        entropy = self.compute_entropy(pre)

        for i in range(pre.shape[0]):
            _, label = torch.max(pre[i], dim=0)
            self.supports[label].append(pre[i])
            self.supports_entropy[label].append(entropy[i].item())

    @staticmethod
    def compute_entropy(x: torch.tensor) -> torch.tensor:
        '''

        :param x: N, D
        :return: N
        '''
        return (F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(1)
