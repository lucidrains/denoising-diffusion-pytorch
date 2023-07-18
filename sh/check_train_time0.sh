for num_samples in 5
do
for diffusion_time_steps in 300 2000
do
for train_num_steps in 5000 10000 20000
do
CUDA_VISIBLE_DEVICES=0 /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
--num_samples ${num_samples} \
--diffusion_time_steps ${diffusion_time_steps} \
--train_num_steps ${train_num_steps} \
--ignore_wandb
done 
done
done