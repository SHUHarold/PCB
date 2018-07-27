#!/bin/sh
python train_mub_test.py --data-dir ../Person_reID_baseline_pytorch/dataset/Market-1501-v15.09.15/ --MUB --max-epoch 60 --train-batch 32 --test-batch 32 --stepsize 20 --eval-step 20 --save-dir logs/resnet50-xent-mub-market1501 --gpu-ids 0,1 --train-all --train-log log_train.txt
