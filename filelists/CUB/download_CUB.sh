#!/usr/bin/env bash
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1
tar -zxvf CUB_200_2011.tgz
python write_CUB_filelist.py
