#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: graphics.py
Description: 
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""

from main import pipeline, delete_saved_objects
import argparse
import numpy as np
import ipdb

def generate_openfield_shapes(input_circle_shape,input_shape_string,percent=0.75):
    """ Generate openfield shapes """
    # Break down components of input shapes
    X,Y,R = input_circle_shape[0][0]
    stringoh = input_shape_string[0][0]

    # Get output shape components for smaller inner circle
    Rnew=np.round(R*percent).astype(np.uint8)

    # Re build output lists
    output_circle_shape = [X,Y,Rnew]
    input_circle_shape = [X,Y,R]
    input_shape_string = stringoh
    output_shape_string = stringoh
    return [[input_circle_shape,output_circle_shape]],[[input_shape_string,output_shape_string]]

if __name__=='__main__':
    # Parse command line inputs
    parser=argparse.ArgumentParser()
    parser.add_argument('--root_directory',type=str,required=True)
    args=parser.parse_args()

    # Delete previously made objects
    delete_saved_objects(root_dir=args.root_directory)

    # Set up main object 
    primaryobject=pipeline(root_dir=args.root_directory)

    # set shapes
    shapesoh,shapestringsoh = generate_openfield_shapes(input_circle_shape=[[[360,260,200]]],input_shape_string=[[['circle']]])
    primaryobject.set_shapes(shape_positions=shapesoh,shapes=shapestringsoh)

    # Run main object
    primaryobject()