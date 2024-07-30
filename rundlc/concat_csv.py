#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:21:19 2022

@author: dje4001
"""
import os,glob
import pandas as pd
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--video_root_dir',type=str,required=True)
parser.add_argument('--final_filename',type=str,required=True)

if __name__=="__main__":    
    args=parser.parse_args()
    os.chdir(args.video_root_dir)
    all_files=[i for i in glob.glob('*.csv')]
    combined_csv=pd.concat([pd.read_csv(f) for f in all_files])
    combined_csv.to_csv(args.final_filename,index=False)
