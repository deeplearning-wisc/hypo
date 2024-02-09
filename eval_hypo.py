import argparse
import math
import os
import time
from datetime import datetime
import logging
import tensorboard_logger as tb_logger
import pprint

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import numpy as np


from make_datasets_cifar import *

from sklearn.metrics import accuracy_score

from utils import (CompLoss, DisLoss, DisLPLoss, set_loader_small, set_loader_ImageNet, set_model)

parser = argparse.ArgumentParser(description='Eval HYPO')
parser.add_argument('--gpu', default=7, type=int, help='which GPU to use')
parser.add_argument('--seed', default=4, type=int, help='random seed')  # original 4
parser.add_argument('--w', default=2, type=float,
                    help='loss scale')
parser.add_argument('--proto_m', default= 0.99, type=float,
                   help='weight of prototype update')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset', choices=['PACS', 'CIFAR-10', 'ImageNet-100'])
parser.add_argument('--id_loc', default="datasets/CIFAR10", type=str, help='location of in-distribution dataset')
parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, wrt40, wrt28, densenet100, resnet50, resnet34]')
parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
parser.add_argument('--loss', default = 'hypo', type=str, choices = ['hypo'],
                    help='name of experiment')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
parser.add_argument('--save-epoch', default=100, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default= 128, type=int, 
                    help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', default=0.5, type=float,
                    help='initial learning rate')
# if linear lr schedule
parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180',
                        help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
# if cosine lr schedule
parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
parser.add_argument('--normalize', action='store_true',
                        help='normalize feat embeddings')

parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--target_domain', type=str, default='sketch', choices=['sketch', 'photo', 'art_painting', 'cartoon'])
parser.add_argument('--cortype', type=str, default='gaussian_noise', help='data type of corrupted datasets')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()


state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%d_%m_%H:%M")

#processing str to list for linear lr scheduling
args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]

args.name = (f"{date_time}_hypo_{args.model}_lr_{args.learning_rate}_cosine_"
    f"{args.cosine}_bsz_{args.batch_size}_td_{args.target_domain}_head_{args.head}_{args.loss}_wd_{args.w}_{args.epochs}_{args.feat_dim}_"
    f"trial_{args.trial}_temp_{args.temp}_{args.in_dataset}_pm_{args.proto_m}")

args.log_directory = "logs/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name= args.name)
args.model_directory = "checkpoints/hypo/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name= args.name )
args.tb_path = './save/hypo/{}_tensorboard'.format(args.in_dataset)
if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)
args.tb_folder = os.path.join(args.tb_path, args.name)
if not os.path.isdir(args.tb_folder):
    os.makedirs(args.tb_folder)

#save args
with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

#init log
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(message)s')
fileHandler = logging.FileHandler(os.path.join(args.log_directory, "train_info.log"), mode='w')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.addHandler(streamHandler) 

log.debug(state)

if args.in_dataset == "CIFAR-10":
    args.n_cls = 10
elif args.in_dataset == "PACS":
    args.n_cls = 7
elif args.in_dataset == "VLCS":
    args.n_cls = 5
elif args.in_dataset == "OfficeHome":
    args.n_cls = 65
elif args.in_dataset == 'terra_incognita':
    args.n_cls = 10
elif args.in_dataset in ["CIFAR-100", "ImageNet-100"]:
    args.n_cls = 100


#set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
log.debug(f"{args.name}")

# warm-up for large-batch training
if args.batch_size > 256:
    args.warm = True
if args.warm:
    args.warmup_from = 0.001
    args.warm_epochs = 10
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.learning_rate

        
def to_np(x): return x.data.cpu().numpy()


if args.in_dataset == 'CIFAR-10':
    val_loader, test_loader_ood = make_datasets(args.in_dataset, state, args.cortype)

else:
    train_loader, val_loader, test_loader_ood = set_loader_small(args)


print("\n len(loader_in.dataset) {}, " \
    "len(test_loader_ood.dataset) {}".format(
    len(val_loader.dataset),
    len(test_loader_ood.dataset)))


def main():
    tb_log = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    model = set_model(args)

    model_name='./checkpoint_max.pth.tar'
    model.load_state_dict(torch.load(model_name)['state_dict'])

    criterion_comp = CompLoss(args, temperature=args.temp).cuda()
    # V1: learnable prototypes
    # criterion_dis = DisLPLoss(args, model, val_loader, temperature=args.temp).cuda() # V1: learnable prototypes
    # optimizer = torch.optim.SGD([ {"params": model.parameters()},
    #                               {"params": criterion_dis.prototypes}  
    #                             ], lr = args.learning_rate,
    #                             momentum=args.momentum,
    #                             nesterov=True,
    #                             weight_decay=args.weight_decay)

    # V2: EMA style prototypes
    criterion_dis = DisLoss(args, model, val_loader, temperature=args.temp).cuda() # V2: prototypes with EMA style update

    criterion_dis.load_state_dict(torch.load(model_name)['dis_state_dict'])

    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)


    model.eval()


    print("computing over distribution ID dataset. \n")
    with torch.no_grad():
        accuracies_in = []
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()

            penultimate = model.encoder(data).squeeze()
            penultimate = F.normalize(penultimate, dim=1)

            features = model.forward(data)

            feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)

            # for numerical stability
            logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
            logits = feat_dot_prototype - logits_max.detach()

            pred = logits.data.max(1)[1]
            accuracies_in.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

    acc = sum(accuracies_in) / len(accuracies_in)
    print("ID accuracy: {}".format(acc))

    print("computing over test distribution cor dataset. \n")
    with torch.no_grad():
        accuracies_cor = []
        for data, target in test_loader_ood:
            data, target = data.cuda(), target.cuda()

            penultimate = model.encoder(data).squeeze()
            penultimate = F.normalize(penultimate, dim=1)

            features = model.forward(data)

            feat_dot_prototype = torch.div(torch.matmul(features, criterion_dis.prototypes.T), args.temp)

            # for numerical stability
            logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
            logits = feat_dot_prototype - logits_max.detach()

            pred = logits.data.max(1)[1]
            accuracies_cor.append(accuracy_score(list(to_np(pred)), list(to_np(target))))

    acc_cor = sum(accuracies_cor) / len(accuracies_cor)

    if args.in_dataset == 'CIFAR-10':  
        print("OOD accuracy for generalization: {}, corrupted types is: {}".format(acc_cor, args.cortype))
    else:
        print("OOD accuracy for generalization: {}, target domain is: {}".format(acc_cor, args.target_domain))




if __name__ == '__main__':
    main()
