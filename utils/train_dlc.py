#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DLC model when given config pathway.
"""
import argparse
import deeplabcut

def train_dlc(path_to_config):
    path_to_config=str(path_to_config)
    deeplabcut.load_demo_data(path_to_config)
    deeplabcut.train_network(path_to_config, shuffle=1,displayiters=5,saveiters=100)
    deeplabcut.evaluate_network(path_to_config)


parser=argparse.ArgumentParser()
parser.add_argument('--config_file_dir',type=str,required=True) #input to config file

if __name__=="__main__":
    args=parser.parse_args()
    train_dlc(args.config_file_dir)
