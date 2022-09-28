#!/bin/sh
#SBATCH -J ENS6PS      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/ENS6PS-sweep-%j.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/ENS6PS-sweep-%j.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
wandb agent --count 5 szztn951357/ERA5_20-12/bkvtjdfw
