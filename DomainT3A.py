'''
this is a improvement of T3A.
Huanran Chen came up with this idea, which use domain label to improve the generalization of T3A

Huanran Chen
2022 06 29
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class T3A(nn.Module):
    def __init__(self, last_layer: nn.Linear, k=0, num_domain=6, num_classes=60, in_dim=864):
        '''

        :param last_layer:
        :param k: select k max when get features. if k <=0, select all.
        '''
        super(T3A, self).__init__()
        self.last_layer = last_layer
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.supports = {}
        self.supports_entropy = {}
        self.supports_label = {}
        self.num_domain = num_domain
        self.num_classes = num_classes

        # each domain is a list
        for i in range(num_domain):
            self.supports[i] = []
            self.supports_entropy[i] = []
            self.supports_label[i] = []

        self.domain_classifier = nn.Linear(in_dim, num_domain).to(self.device)

        self.geted_supports = None  # tensor (num_domain , num_classes , D)
        # attention!! you can only get support once!!!!!!!!!#

    def forward(self, x, adapt=False, use_T3A=False, domain_label=None):
        '''

        :param x:
        :param adapt: whether use this batch to update parameter of this model
        :param use_T3A: whether use T3A to predict or use nn.Linear to predict
        :return: predict N, num_classes
        '''
        if adapt and domain_label is not None:
            self.update_supports(x, domain_label)
        if not use_T3A and self.training:
            # if training, return domain logits to train domain classifier
            return self.last_layer(x), self.domain_classifier(x)
        if not use_T3A and not self.training:
            return self.last_layer(x)

        # use T3A
        if self.geted_supports is None:
            weight = self.get_supports().permute(1, 0)  # D, num_classes
        else:
            weight = self.geted_supports
        bias = self.last_layer.bias.data
        # num_domain , num_classes , D
        weight = weight.permute(0, 2, 1)  # num_domain , D , num_classes
        logits = x @ weight + bias  # num_domain, N, num_classes
        logits = logits.permute(1, 2, 0)  # N, num_clases, num_domain
        domain_prob = torch.softmax(self.domain_classifier(x), dim=1).unsqueeze(2)  # N,num_domain, 1
        logits = torch.bmm(logits, domain_prob).squeeze()
        return logits

    def get_supports(self, k=None) -> torch.tensor:
        '''

        :param k:
        :return: num_domain , num_classes , D
        '''
        if k is None:
            k = self.k
        weight = []

        for domain in range(self.num_domain):
            # in each iteration, get a (num_classes, D) tensor into weight.
            # and finally use torch.cat to get return tensor
            now_weight = []
            now_domain_support = torch.cat(self.supports[domain], dim=0)  # N, D
            # N
            now_domain_support_label = torch.tensor(self.supports_label[domain], device=self.device)
            # N
            now_domain_support_entropy = torch.tensor(self.supports_entropy[domain], device=self.device)
            for now_class in range(self.num_classes):
                class_mask = now_domain_support_label == now_class
                all = now_domain_support[class_mask, :]
                if k <= 0:
                    pass
                else:
                    all_entropy = now_domain_support_entropy[class_mask]
                    _, indices = torch.sort(all_entropy, dim=0, descending=False)  # from small to big
                    all = all[indices, :]
                    all = all[torch.arange(k - 1), :]

                now_weight.append(all.mean(0))

            now_weight = torch.cat(now_weight, dim=0)  # num_classes, D
            weight.append(now_weight)

        weight = torch.stack(weight)
        self.geted_supports = weight
        # num_domain , num_classes , D
        return weight

    def update_supports(self, x, domain_label):
        '''

        :param x: N, D
        :param domain_label: N,
        :return:
        '''
        pre = self.last_layer(x)  # N, D
        entropy = self.compute_entropy(pre)

        for i in range(pre.shape[0]):
            now_domain = domain_label[i].item()
            _, now_pre = torch.max(pre, dim=1)
            self.supports[now_domain].append(pre[i])
            self.supports_entropy[now_domain].append(entropy[i].item())
            self.supports_label[now_domain].append(now_pre.item())

    @staticmethod
    def compute_entropy(x: torch.tensor) -> torch.tensor:
        '''

        :param x: N, D
        :return: N
        '''
        return - (F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(1)



