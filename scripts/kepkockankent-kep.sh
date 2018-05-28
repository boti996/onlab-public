#!/usr/bin/env bash
input=$1
mkdir output
ffmpeg -i $input -vf fps=1 ./output/out%d.png

