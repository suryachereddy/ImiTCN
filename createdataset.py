from tkinter.tix import IMAGE

from click import argument
from util import (SingleViewTripletBuilder, distance, Logger, ensure_folder)
import argparse
import numpy as np
from tqdm import tqdm 
import torch

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
    arguments=get_args()
    samplesize=200
    triplet_builder = SingleViewTripletBuilder("./data/validation/", IMAGE_SIZE, arguments, sample_size=200)
    videos=triplet_builder.video_count
    print(videos)
    for i in tqdm(range((videos*5)//samplesize)):
        
        
        dataset = triplet_builder.build_set()
        torch.save(dataset,"./data/validation_triplets/triplets_"+str(i))



if __name__ == '__main__':
    main()