import torch
import torch.nn as nn
import torch.optim as optim
import loss
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from util import save_log
import lfw_verification
from loader import get_loader


def Make_layer(block, num_filters, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block(num_filters))
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def spherenet(num_layers, feature_dim, use_pool, use_dropout):
    """SphereNets.
    We follow the paper, and the official caffe code:
        SphereFace: Deep Hypersphere Embedding for Face Recognition, CVPR, 2017.
        https://github.com/wy1iu/sphereface
    """
    class SphereResBlock(nn.Module):
        def __init__(self, channels):
            super(SphereResBlock, self).__init__()
            self.resblock = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.PReLU(channels)
            )

        def forward(self, x):
            return x + self.resblock(x)

    filters = [64, 128, 256, 512]
    if num_layers == 4:
        units = [0, 0, 0, 0]
    elif num_layers == 10:
        units = [0, 1, 2, 0]
    elif num_layers == 20:
        units = [1, 2, 4, 1]
    elif num_layers == 36:
        units = [2, 4, 8, 2]
    elif num_layers == 64:
        units = [3, 8, 16, 3]
    net_list = []
    for i, (num_units, num_filters) in enumerate(zip(units, filters)):
        if i == 0:
            net_list += [nn.Conv2d(3, 64, 3, 2, 1), nn.PReLU(64)]
        elif i == 1:
            net_list += [nn.Conv2d(64, 128, 3, 2, 1), nn.PReLU(128)]
        elif i == 2:
            net_list += [nn.Conv2d(128, 256, 3, 2, 1), nn.PReLU(256)]
        elif i == 3:
            net_list += [nn.Conv2d(256, 512, 3, 2, 1), nn.PReLU(512)]
        if num_units > 0:
            net_list += [Make_layer(SphereResBlock, num_filters=num_filters, num_of_layer=num_units)]
    if use_pool:
        net_list += [nn.AdaptiveAvgPool2d((1, 1))]
    net_list += [Flatten()]
    if use_dropout:
        net_list += [nn.Dropout()]
    if use_pool:
        net_list += [nn.Linear(512, feature_dim)]
    else:
        net_list += [nn.Linear(512*7*6, feature_dim)]
    return nn.Sequential(*net_list)


def deep_face_recognition(args):

    log = args.log_path

    dataloader = get_loader('webface', args.bs)
    class_num = dataloader.num_class

    gpu_ids = args.gpu_ids
    device = args.device
    message = '*' * 40 + '\n' + \
              'Face recognition by Spherenet20 based on {}...'.format(args.loss_type) + '\n' + \
              'Running on {}'.format(device)
    save_log(message, log)

    # Setup network Spherenet20
    num_layers = 20
    backbone = spherenet(num_layers, args.feature_dim, args.use_pool, args.use_dropout)
    backbone.to(device)

    # Setup nn.DataParallel if necessary
    if len(gpu_ids) > 1:
        backbone = nn.DataParallel(backbone)

    # Objective function
    criterion = getattr(loss, args.loss_type)
    criterion = criterion(class_num=class_num, args=args)
    criterion.to(device)

    # Setup optimizer
    lr = args.lr
    params = list(backbone.parameters()) + list(criterion.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    ## Set decay steps
    str_steps = args.decay_steps.split(',')
    args.decay_steps = []
    for str_step in str_steps:
        str_step = int(str_step)
        args.decay_steps.append(str_step)
    scheduler = MultiStepLR(optimizer, milestones=args.decay_steps, gamma=0.1)

    # ------------------------
    # Start Training and Validating
    # ------------------------
    pbar = tqdm(range(1, args.iterations + 1), ncols=0)
    for total_steps in pbar:
        backbone.train()
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        input, target = dataloader.next()
        features = backbone(input.to(device))
        score, loss_ce = criterion(features, target.to(device))

        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        if total_steps % args.eval_freq == 0:
            backbone.eval()
            message = '{}: {:.4f} , {}: {:.4f}'.format('loss_ce', loss_ce, 'lr', lr)
            save_log(message, log)
            lfw_verification.run(backbone, args, total_steps)
        # display
        description = '{}: {:.4f} , {}: {:.4f}'.format('loss_ce', loss_ce, 'lr', lr)
        pbar.set_description(desc=description)
    torch.save(backbone.state_dict(), args.model_root)