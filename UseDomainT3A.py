# example of domain T3A
import torch
import torch.nn as nn
import torch.nn.functional as F
from DomainT3A import T3A
from tqdm import tqdm
from torch.utils.data import DataLoader


def use_domainT3A(train_loader: DataLoader, test_loader: DataLoader,
                  model: nn.Module, num_epochs=1, TTA_epoch=0,
                  need_train = False):
    '''

    :param train_loader: to train domain_classifier
    :param test_loader:
    :param model:
    :param num_epochs:
    :param TTA_epoch:
    :return:
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = nn.Sequential(*list(model.children())[:-1])
    features.requires_grad_(False)
    last_layer = list(model.children())[-1]
    t3a = T3A(last_layer=last_layer).to(device)
    
    if need_train:
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(t3a.parameters(), lr=1e-3)

        print('now we start training')
        print('-' * 100)
        t3a.train()
        for epoch in range(1, num_epochs + 1):
            pbar = tqdm(train_loader)
            for step, (x, y, domain) in enumerate(pbar):
                x = x.to(device)
                domain = domain.to(device)
                with torch.no_grad():
                    x = features.forward(x)
                    x = x.squeeze()
                _, x = t3a.forward(x, adapt=True, use_T3A=False, domain_label=domain)
                loss = criterion(x, domain)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        t3a.save_supports()
    else:
        t3a.load_supports()
    # predict
    t3a.eval()
    t3a.get_supports(10)
    result = {}
    print('now we start predicting')
    with torch.no_grad():
        for x, names in tqdm(test_loader):
            x = x.to(device)
            x = features(x)
            x = t3a.forward(x, adapt=False, use_T3A=True, domain_label=None)  # N, D
            _, x = torch.max(x, dim=1)  # N,
            for i, name in enumerate(list(names)):
                result[name] = x[i].item()

    from data.dataUtils import write_result
    write_result(result)
    print('finished!!')


if __name__ == '__main__':
    train_image_path = './public_dg_0416/train/'
    valid_image_path = './public_dg_0416/train/'
    label2id_path = './dg_label_id_mapping.json'
    test_image_path = './public_dg_0416/public_test_flat/'
    from data.data import get_loader, get_test_loader

    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument('-b', '--batch_size', default=512)
    paser.add_argument('-t', '--total_epoch', default=1)
    paser.add_argument('-l', '--lr', default=1e-3)
    args = paser.parse_args()
    batch_size = int(args.batch_size)
    total_epoch = int(args.total_epoch)
    lr = float(args.lr)

    train_loader = get_loader(batch_size=batch_size,
                              valid_category=None,
                              train_image_path=train_image_path,
                              valid_image_path=valid_image_path,
                              label2id_path=label2id_path)
    test_loader_predict, _ = get_test_loader(batch_size=batch_size,
                                             transforms=None,
                                             label2id_path=label2id_path,
                                             test_image_path=test_image_path)
    test_loader_student, label2id = get_test_loader(batch_size=batch_size,
                                                    transforms='train',
                                                    label2id_path=label2id_path,
                                                    test_image_path=test_image_path)
    from pyramidnet import pyramidnet272
    model = pyramidnet272(num_classes=60)
    model.load_state_dict(torch.load('KD.pth'))
    model.to(torch.device('cuda'))

    use_domainT3A(train_loader, test_loader_predict, model, num_epochs=total_epoch)
