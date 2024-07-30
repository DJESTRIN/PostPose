#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create DLC project
"""
import deeplabcut
import glob

root_dir='/athena/listonlab/store/dje4001/bb0046_temp_store/**/*.mp4'
files=glob.glob(root_dir,recursive=True)

deeplabcut.create_new_project('TMT','Estrin',files,working_directory='/athena/listonlab/scratch/dje4001/mdt02/tmt_dlc_class/',copy_videos=False,multianimal=False)
