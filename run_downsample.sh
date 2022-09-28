#!/bin/sh
#SBATCH -J downsample      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/downsample.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/downsample.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;

python downsamplefrom720.py