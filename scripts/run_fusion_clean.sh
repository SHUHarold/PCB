#!/bin/bash
python train.py --gpu_ids 0 --train_all --use_clean_imgs --batchsize 32 --PCB --name PCB-32-clean --data_dir ../Alg-VideoAlgorithm/re-ID/Person_reID_baseline_pytorch/dataset
