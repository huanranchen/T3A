# T3A
unofficial pytorch implement of Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization


some improvement:
1. use dictionary to save supports in order to save CUDA memory
2. keep bias and use it.

From my point of view, weight of last linear layer = features of each class, bias is a kind of statistic that counts the probability of occurrence of various types
So, I opt to keep bias.








domainT3A is a improvement of T3A proposed by me, which use domain label when predict

P(y|x) = P(y|x,D)*P(D|x)
