
for batch_size in 4 16 
do
for accumulation_steps in 1 4
do 
for GDMod_type in NGmod_absoluteNone NGmod_absolute
do   
name=${GDMod_type:13:2}${batch_size}.${accumulation_steps}.7066N
echo $name
sbatch -p ai4science -N1 -c16 --gres=gpu:1 -J $name -o log/%j-$name.out -e log/%j-$name.out run.sh $GDMod_type $accumulation_steps $batch_size
done 
done
done
