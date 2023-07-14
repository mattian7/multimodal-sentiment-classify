import torch
import argparse
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import numpy as np
from model import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(3047)
torch.cuda.manual_seed(3047)
torch.cuda.manual_seed_all(3047)
np.random.seed(3047)


def parse_arguments():
    parser = argparse.ArgumentParser(description="AI5-HOMEWORK")
    parser.add_argument("--mode", type=str, choices=["text", "image", "all"], default="all")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument("--test", type=int, choices=[0, 1], default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--fusion_dropout', type=float, default=0.5)
    parser.add_argument('--text_dropout', type=float, default=0.1)
    parser.add_argument('--image_dropout', type=float, default=0.1)
    parser.add_argument('--fusion_dim', type=int, default=128)
    parser.add_argument('--text_dim', type=int, default=64)
    parser.add_argument('--image_dim', type=int, default=32)
    args = parser.parse_args()
    return args


def update_center(modal, label):
    hidden_size = modal.shape[1]
    # print(modal.shape)
    mask0 = (label == 0)
    mask1 = (label == 1)
    mask2 = (label == 2)
    # print(mask0.shape)
    d0 = torch.masked_select(modal, mask0.unsqueeze(1))
    d1 = torch.masked_select(modal, mask1.unsqueeze(1))
    d2 = torch.masked_select(modal, mask2.unsqueeze(1))
    # 计算平均值
    center0 = torch.mean(d0.view(-1, hidden_size), dim=0)
    center1 = torch.mean(d1.view(-1, hidden_size), dim=0)
    center2 = torch.mean(d2.view(-1, hidden_size), dim=0)

    return center0, center1, center2


def update_labels(modal, center0, center1, center2):
    input_tensor = modal

    a_exp = center0.expand_as(input_tensor)
    b_exp = center1.expand_as(input_tensor)
    c_exp = center2.expand_as(input_tensor)

    distances = torch.norm(input_tensor - torch.stack([a_exp, b_exp, c_exp]), dim=2)

    labels = torch.argmin(distances, dim=0)
    return labels


def model_train(args):
    """训练模型并保存至./model.pth"""

    train_data_list, test_data_list = get_data_list()
    train_data_list, test_data_list = data_preprocess(train_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
    model = SELF_MM(args)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(lr=args.lr, params=optimizer_grouped_parameters)
    criterion = CrossEntropyLoss()
    best_rate = 0
    print('[START_OF_TRAINING_STAGE]')
    '''
    text_center0 = torch.zeros((16, 768))
    text_center1 = torch.zeros((16, 768))
    text_center2 = torch.zeros((16, 768))

    image_center0 = torch.zeros((16, 2048))
    image_center1 = torch.zeros((16, 2048))
    image_center2 = torch.zeros((16, 2048))
    '''
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.train()
        for idx, (guid, tag, image, text) in enumerate(train_data_loader):
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            if args.mode == "text":
                out = model(None, text)
                loss = criterion(out["fusion_output"], tag)
            elif args.mode == "image":
                out = model(image, None)
                loss = criterion(out["fusion_output"], tag)
            else:
                out = model(image, text)
                if idx >= 1 | epoch >= 1:
                    label_t = update_labels(out["text_state"], text_center0, text_center1, text_center2)
                    label_m = update_labels(out["image_state"], image_center0, image_center1, image_center2)
                else:
                    label_t = tag
                    label_m = tag

                text_center0, text_center1, text_center2 = update_center(out["text_state"], label_t)
                image_center0, image_center1, image_center2 = update_center(out["image_state"], label_m)

                loss = criterion(out["fusion_output"], tag) + criterion(out["text_output"], label_t) + criterion(
                    out["image_output"], label_m)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(guid)
            pred = torch.max(out["fusion_output"], 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total

        rate = correct / total * 100
        print("Epoch {} --- Loss {:.5f} --- Acc {:.2f}%".format(epoch + 1, total_loss, rate))
        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.eval()

        for guid, tag, image, text in valid_data_loader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)

            if args.mode == "text":
                out = model(None, text)
            elif args.mode == "image":
                out = model(image, None)
            else:
                out = model(image, text)

            # loss = criterion(out, tag)

            # total_loss += loss.item() * len(guid)
            pred = torch.max(out["fusion_output"], 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        # total_loss /= total
        rate = correct / total * 100
        print("Valid Acc {:.2f}%".format(rate))
        if (rate > best_rate) & (args.mode == "all"):
            best_rate = rate
            print('Save best result on valid acc {:.2f}%'.format(rate))
            torch.save(model.state_dict(), 'checkpoints/model.pth')
        print()
    print('[END_OF_TRAINING_STAGE]')


def model_test(args):
    train_data_list, test_data_list = get_data_list()
    train_data_list, test_data_list = data_preprocess(train_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
    model = SELF_MM(args)
    model.load_state_dict(torch.load('checkpoints/model.pth'))
    model.to(device)
    print('Test Start')
    guid_list = []
    pred_list = []
    model.eval()

    for guid, tag, image, text in test_data_loader:
        image = image.to(device)
        text = text.to(device)

        if args.mode == "text":
            out = model(None, text)
        elif args.mode == "image":
            out = model(image, None)
        else:
            out = model(image, text)

        pred = torch.max(out["fusion_output"], 1)[1]
        guid_list.extend(guid)
        pred_list.extend(pred.cpu().tolist())

    pred_mapped = {
        0: 'negative',
        1: 'neutral',
        2: 'positive',
    }
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guid_list, pred_list):
            f.write(f'{guid},{pred_mapped[pred]}\n')
        f.close()
        print('prediction has been written to result.txt')
    print('Test End')


if __name__ == "__main__":
    args = parse_arguments()
    if args.test == 1:
        model_test(args)
    else:
        model_train(args)
