for dataset_flag in  3D70N #3D70N
do
for flag in train #test valid 
do
    nohup python test.py $dataset_flag $flag > log/offlinedata.$dataset_flag.$flag.2&
done
done