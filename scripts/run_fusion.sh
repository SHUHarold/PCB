#!/bin/bash
python train.py --gpu_ids 0 --train_all --batchsize 32 --PCB --name PCB-32 --data_dir ../Alg-VideoAlgorithm/re-ID/Person_reID_baseline_pytorch/dataset
