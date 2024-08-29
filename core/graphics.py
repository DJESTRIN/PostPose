#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: graphics.py
Description: 
Author: David Estrin
Version: 1.0
Date: 08-29-2024
"""

class graphics():
    def __init__(self,digested_obj,output_directory,video_file=[]):
        # Did user include a video for us to use for plotting?
        if video_file:
            self.video_file=video_file
            self.attached_video=True
        else:
            self.attached_video=True

    def __call__(self):
        # Need to code these in later
        self.plot_trajectory()
        self.plot_distance()
        self.plot_speed()
        self.plot_acceleration()