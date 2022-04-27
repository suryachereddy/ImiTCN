from asyncore import read
from tkinter.tix import IMAGE

from click import argument
from util import (SingleViewTripletBuilder, distance, Logger, ensure_folder,read_video)
import argparse
import numpy as np
from tqdm import tqdm 
import torch
import random
IMAGE_SIZE = (299, 299)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--train-directory', type=str, default='./data/train/')
    parser.add_argument('--validation-directory', type=str, default='./data/validation/')
    parser.add_argument('--minibatch-size', type=int, default=32)
    parser.add_argument('--margin', type=float, default=10.0)
    parser.add_argument('--model-name', type=str, default='tcn')
    parser.add_argument('--log-file', type=str, default='./out.log')
    parser.add_argument('--lr-start', type=float, default=0.01)
    parser.add_argument('--triplets-from-videos', type=int, default=5)
    return parser.parse_args()



def main():
    #arguments=get_args()
    samplesize=2
    max=80
    idx=[i for i in range(80)]

    random.shuffle(idx)
    i=0
    for _ in tqdm(range(40)):
        sample1=idx.pop()
        sample2=idx.pop()
        avideo = read_video(f"./regressiondata/video_{sample1}.npy", (299,299))
        ajoint = np.load(f"./regressiondata/joint_{sample1}.npy")
        bvideo = read_video(f"./regressiondata/video_{sample2}.npy", (299,299))
        bjoint = np.load(f"./regressiondata/joint_{sample2}.npy")

        np.save(f'./regressiondataprocessed/x_{i}.npy', np.vstack((avideo,bvideo)).astype(np.float32))
        np.save(f'./regressiondataprocessed/y_{i}.npy', np.vstack((ajoint,bjoint)).astype(np.float32))
        i+=1



if __name__ == '__main__':
    main()