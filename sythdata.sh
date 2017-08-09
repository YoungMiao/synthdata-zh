#!/usr/bin/env bash

python gen_cn_txt.py \
	--bgs=./bgs_1 \
	--fonts=./fonts_cn  \
	--fh=48 \
	--output=./datasets \
	--label=./word.txt  \
	--trainlabel=./train.txt \
	--vallabel=./val.txt \
	--sumnumber=4 \
	--trainnum=3 
