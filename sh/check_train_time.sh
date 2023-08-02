for num_samples in 1
do
for train_num_steps in 3
do
for sampling_batch_size in 5
do
for diffusion_time_steps in 300
do
CUDA_VISIBLE_DEVICES=0 /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
--num_samples ${num_samples} \
--train_num_steps ${train_num_steps} \
--sampling_batch_size ${sampling_batch_size} \
--diffusion_time_steps ${diffusion_time_steps} \
--save_seq \
--ignore_wandb
done 
done
done
done

