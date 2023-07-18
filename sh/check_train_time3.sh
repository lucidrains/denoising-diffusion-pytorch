for num_samples in 10
do
for diffusion_time_steps in 700 1000
do
for train_num_steps in 5000 10000 20000
do
CUDA_VISIBLE_DEVICES=3 /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
--num_samples ${num_samples} \
--diffusion_time_steps ${diffusion_time_steps} \
--train_num_steps ${train_num_steps}
done 
done
done