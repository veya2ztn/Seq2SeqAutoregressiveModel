#!/bin/sh
#SBATCH -J createmultitb     # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-createmultitb.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-createmultitb.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python mytool.py --paths checkpoints/WeathBench32x64/ --level 3 --mode createmultitb