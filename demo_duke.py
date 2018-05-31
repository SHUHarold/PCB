import cv2
import os
import scipy.io as sio
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import ft_net, ft_net_dense, PCB, PCB_test

def get_feature(im_cv, model, use_gpu):
    features = torch.FloatTensor()
    x = cv2.resize(im_cv, (192, 384)).astype(np.float32)
    x /= 255.0
    x -= (0.406, 0.456, 0.485)
    x /= (0.225, 0.224, 0.229)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    ff = torch.FloatTensor(1, 2048, 6).zero_()
    for i in range(2):
        if (i == 1):
            x = fliplr(x)
        if use_gpu:
            input_image = Variable(x.cuda())
        else:
            input_image = Variable(x)
        outputs = model(input_image)
        f = outputs.data.cpu()
        ff = ff + f
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    ff = ff.view(ff.size(0), -1)
    features = torch.cat((features, ff), 0)
    return features


def get_reid(model_path):
    checkpoint = torch.load(model_path)
    model_structure = PCB(1453)
    model_structure.load_state_dict(checkpoint)
    model = PCB_test(model_structure)
    model = model.eval()
    return model

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

if __name__ == '__main__':
    root_path = '/workspace/run/project/re-id/Person_reID_baseline_pytorch/dataset/DukeMTMC-reID/train_all/'
    model_path = './model/net_fusion_last.pth'
    model = get_reid(model_path)
    model = model.cuda()
    img_ids = os.listdir(root_path)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    use_gpu = torch.cuda.is_available()
    scores = {}
    store_ids = {}
    if '.DS_Store' in img_ids:
        img_ids.remove('.DS_Store')
    for index, id in enumerate(img_ids):
        features = []
        score = []
        img_root_path = root_path + id
        img_files = os.listdir(img_root_path)
        if '.DS_Store' in img_files:
            img_files.remove('.DS_Store')
        for i in img_files:
            img_path = img_root_path + '/' + i
            im_cv = cv2.imread(img_path)
            features.append(get_feature(im_cv, model, use_gpu))
           # print('--------')
        sum_feature = sum(features)
        mean_feature = sum_feature/len(features)
        for i, feature in  enumerate(features):
            score.append(cos(feature, mean_feature))

        # for index, feature in enumerate(features):
        #     if index == len(features) - 1:
        #         break
        #     for i in range(index + 1, len(features)):
        #         score.append(cos(feature, features[i]))
        scores[id] = score
        store_ids[id] = img_files

        print('----', str(index), '----')
    if not os.path.isdir('./result'):
        os.mkdir('./result')
   # sio.savemat('./result/duke_result.mat', {'scores':scores, 'store_ids':store_ids})
    with open('./result/duke_result.pkl','wb') as f:
        pickle.dump({'scores':scores, 'store_ids':store_ids},f)
    print('---------')
    print('Save Done!')
