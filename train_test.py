# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys
import os
import time
import json
import IPython
import datetime
import argparse
import numpy as np
#import matplotlib
#matplotlib.use('agg')
import os.path as osp
from PIL import Image
#import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms

from eval_metrics import evaluate_deep
from random_erasing import RandomErasing
from model import ft_net, ft_net_dense, PCB, MUB
from utils import AverageMeter, Logger, save_checkpoint

version = torch.__version__

if int(version[2]) <= 3:
    from torch.autograd import Variable
    flag = True
    print("Now using pytoch version {}".format(version))
else:
    flag = False
    print("Now using pytoch version {}".format(version))
  
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
#parser.add_argument('--gpu_ids',default=None, nargs='+', type=int, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--gpu-ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='resnet50', type=str, help='output model name')
parser.add_argument('--train-data-dir', default='./train', type=str, help='train path')
parser.add_argument('--test-data-dir', default='./test', type=str, help='test path')
parser.add_argument('--train-all', action='store_true', help='use all training data')
parser.add_argument('--use-clean-imgs', action='store_true', help='use cleaned images')
parser.add_argument('--color-jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--erasing-p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use-dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--stepsize', default=40, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--train-only', action='store_true', help="training only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='logs')
parser.add_argument('--train-log', type=str, default='log_train.txt')
parser.add_argument('--test-log', type=str, default='log_test.txt')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
opt = parser.parse_args()


def main():
    print("==========\nArgs:{}\n==========".format(opt))
    train_data_dir = opt.train_data_dir
    test_data_dir = opt.test_data_dir
    name = opt.name
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    use_gpu = torch.cuda.is_available()
    if opt.use_cpu: use_gpu = False

    if not opt.evaluate:
        sys.stdout = Logger(osp.join(opt.save_dir, opt.train_log))
    else:
        sys.stdout = Logger(osp.join(opt.save_dir, opt.test_log))

    if use_gpu:
        print("Currently using GPU {}".format(opt.gpu_ids))
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")
    #str_ids = opt.gpu_ids.split(',')
    #gpu_ids = []
    #for str_id in str_ids:
    #    gid = int(str_id)
    #    if gid >=0:
    #        gpu_ids.append(gid)
    #
    ## set gpu ids
    #if len(gpu_ids)>0:
    #    torch.cuda.set_device(gpu_ids[0])
    #print(gpu_ids[0])

    # Load train Data
    # ---------
    print("==========Preparing trian dataset========")
    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288, 144), interpolation=3),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    if opt.PCB:
        transform_train_list = [
            transforms.Resize((384,192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    if opt.erasing_p>0:
        transform_train_list = transform_train_list + [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

    train_data_transforms = transforms.Compose(transform_train_list)
    train_all = ''
    if opt.train_all:
        if opt.use_clean_imgs:
            train_all = '_all_clean'
        else:
            train_all = '_all'
            print("Using all the train images")

    train_image_datasets = {}
    train_image_datasets['train'] = datasets.ImageFolder(os.path.join(train_data_dir, 'train' + train_all),
                                              train_data_transforms)
    train_dataloaders = {x: torch.utils.data.DataLoader(train_image_datasets[x], batch_size=opt.train_batch,
                                                 shuffle=True, num_workers=4)
                  for x in ['train']}
    dataset_sizes = {x: len(train_image_datasets[x]) for x in ['train']}
    class_names = train_image_datasets['train'].classes
    inputs, classes = next(iter(train_dataloaders['train']))


    ######################################################################
    # Prepare test data
    if not opt.train_only:
        print("========Preparing test dataset========")
        transform_test_list = transforms.Compose([
                transforms.Resize((384,192), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        test_image_datasets = {x: datasets.ImageFolder( os.path.join(test_data_dir,x),transform_test_list) for x in ['gallery','query']}
        test_dataloaders = {x: torch.utils.data.DataLoader(test_image_datasets[x], batch_size=opt.test_batch,
                                                     shuffle=False, num_workers=4) for x in ['gallery','query']}


    print("Initializing model...")
    if opt.use_dense:
        model = ft_net_dense(len(class_names))
    else:
        model = ft_net(len(class_names))
    if opt.PCB:
        model = PCB(len(class_names))
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    start_epoch = opt.start_epoch

    if opt.resume:
        print("Loading checkpoint from '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint)
        start_epoch = checkpoint['epoch']
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    if opt.evaluate:
        print("Evaluate only")
        test(model, test_image_datasets, test_dataloaders, use_gpu)
        return
    criterion = nn.CrossEntropyLoss().cuda()
    if opt.PCB:
        ignored_params = list(map(id, model.module.resnet50.fc.parameters() ))
        ignored_params += (list(map(id, model.module.classifier0.parameters() ))
                         +list(map(id, model.module.classifier1.parameters() ))
                         +list(map(id, model.module.classifier2.parameters() ))
                         +list(map(id, model.module.classifier3.parameters() ))
                         +list(map(id, model.module.classifier4.parameters() ))
                         +list(map(id, model.module.classifier5.parameters() ))
                         #+list(map(id, model.classifier7.parameters() ))
                          )
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = optim.SGD([
                 {'params': base_params, 'lr': 0.01},
                 {'params': model.module.resnet50.fc.parameters(), 'lr': 0.1},
                 {'params': model.module.classifier0.parameters(), 'lr': 0.1},
                 {'params': model.module.classifier1.parameters(), 'lr': 0.1},
                 {'params': model.module.classifier2.parameters(), 'lr': 0.1},
                 {'params': model.module.classifier3.parameters(), 'lr': 0.1},
                 {'params': model.module.classifier4.parameters(), 'lr': 0.1},
                 {'params': model.module.classifier5.parameters(), 'lr': 0.1},
                 #{'params': model.classifier7.parameters(), 'lr': 0.01}
             ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.module.model.fc.parameters())) + list(map(id, model.module.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.01},
            {'params': model.module.model.fc.parameters(), 'lr': 0.1},
            {'params': model.module.classifier.parameters(), 'lr': 0.1}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    # Decay LR by a factor of 0.1 every 40 epochs
    if opt.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=0.1)

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

    for epoch in range(start_epoch, opt.max_epoch):
        start_train_time = time.time()
        if opt.train_only:
            print("==> Training only")
            train(epoch, model, criterion, optimizer, train_dataloaders, use_gpu)
            train_time += round(time.time() - start_train_time)
            if epoch % opt.eval_step == 0:
                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'epoch': epoch,
                }, 0, osp.join(opt.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        
            if opt.stepsize > 0: scheduler.step()
        else:
            train(epoch, model, criterion, optimizer, train_dataloaders, use_gpu)
            train_time += round(time.time() - start_train_time)

            if opt.stepsize > 0: scheduler.step()
            if (epoch+1) > opt.start_eval and opt.eval_step > 0 and (epoch+1) % opt.eval_step == 0 or (epoch+1) == opt.max_epoch:
                print("==> Test")
                rank1 = test(model, test_image_datasets, test_dataloaders, use_gpu)
                is_best = rank1 > best_rank1
                if is_best:
                    best_rank1 = rank1
                    best_epoch = epoch + 1
                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(opt.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
    if not opt.train_only:
        print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, optimizer, train_dataloaders, use_gpu):
    since = time.time()
    losses = AverageMeter()
    # Each epoch has a training and validation phase
    for phase in ['train']:
        model.train(True)  # Set model to training mode
        # Iterate over data.
        for batch_index, data in enumerate(train_dataloaders[phase]):
            # get the inputs
            inputs, labels = data
            #print(inputs.shape)
            # wrap them in Variable
            if flag:
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            else:
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = inputs, labels
                

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            if not opt.PCB:
                loss = criterion(outputs, labels)
            else:
                part = {}
                sm = nn.Softmax(dim=1)
                num_part = 6
                for i in range(num_part):
                    part[i] = outputs[i]
                loss = criterion(part[0], labels)
                for i in range(num_part - 1):
                    loss += criterion(part[i + 1], labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
            if flag:
                losses.update(loss.data[0], labels.size(0))
            else:
                losses.update(loss.item(), labels.size(0))
                
            if (batch_index+1) % opt.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, batch_index+1,len(train_dataloaders[phase]),loss=losses))


######################################################################
# Draw Curve
#---------------------------
#x_epoch = []
#fig = plt.figure()
#ax0 = fig.add_subplot(121, title="loss")
#ax1 = fig.add_subplot(122, title="top1err")
#def draw_curve(current_epoch):
#    x_epoch.append(current_epoch)
#    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
#    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
#    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
#    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
#    if current_epoch == 0:
#        ax0.legend()
#        ax1.legend()
#    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


#dir_name = os.path.join('./model',name)
#if not os.path.isdir(dir_name):
#    os.makedirs(dir_name)
#
## save opts
#with open('%s/opts.json'%dir_name,'w') as fp:
#    json.dump(vars(opt), fp, indent=1)

def test(model, test_image_datasets, test_dataloaders, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()
    queryloader = test_dataloaders['query']
    galleryloader = test_dataloaders['gallery']

    #with torch.no_grad():
    qf, q_pids, q_camids = [], [], []
    q_path = test_image_datasets['query'].imgs
    q_camids,q_pids = get_id(q_path)
    for batch_idx, q_data in enumerate(queryloader):
        imgs, _ = q_data
        if flag:
            imgs = Variable(imgs.cuda())
        else:
            imgs = imgs.cuda()

        end = time.time()
        features = model(imgs)
        features = torch.norm(features, p=2, dim=1, keepdim=True)
        if opt.PCB:
            features = features.view(features.size(0), -1)
        batch_time.update(time.time() - end)

        features = features.data.cpu()
        qf.append(features)
        #q_pids.extend(pids)
        #q_camids.extend(camids)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    g_path = test_image_datasets['gallery'].imgs
    g_camids,g_pids = get_id(g_path)
    end = time.time()
    for batch_idx, g_data in enumerate(galleryloader):
        imgs, _ = g_data
        if flag:
            imgs = Variable(imgs.cuda())
        else:
            imgs = imgs.cuda()

        end = time.time()
        features = model(imgs)
        features = torch.norm(features, p=2, dim=1, keepdim=True)
        if opt.PCB:
            features = features.view(features.size(0), -1)
        batch_time.update(time.time() - end)

        features = features.data.cpu()
        gf.append(features)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, opt.test_batch))
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate_deep(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


if __name__ == '__main__':
    main()
