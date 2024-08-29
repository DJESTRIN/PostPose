#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: concatcsv.py
Author:      David Estrin
"""
import os,glob
import pandas as pd
import argparse

def concatcsv(root_dir,final_filename,extension='*.csv'):
    """ Search for csv files in root directory.
    Find csv files and then concatonate them into a single dataframe.
    For dataframes only.
    """
    os.chdir(root_dir)
    all_files=[i for i in glob.glob(extension)]
    combined_csv=pd.concat([pd.read_csv(f) for f in all_files])
    combined_csv.to_csv(final_filename,index=False)

if __name__=="__main__":    
    parser=argparse.ArgumentParser()
    parser.add_argument('--video_root_dir',type=str,required=True)
    parser.add_argument('--final_filename',type=str,required=True)
    args=parser.parse_args()
    concatcsv(args.video_root_dir,args.final_filename)
  
