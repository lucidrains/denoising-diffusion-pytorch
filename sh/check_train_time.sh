for num_samples in 5
do
for diffusion_time_steps in 300 700 1000 2000
do
for train_num_steps in 5000
do
for sampling_batch_size in 128 512 1024 2048 4096
do
CUDA_VISIBLE_DEVICES=1 /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
--num_samples ${num_samples} \
--diffusion_time_steps ${diffusion_time_steps} \
--train_num_steps ${train_num_steps} \
--sampling_batch_size ${sampling_batch_size} \
--sample_only
done 
done
done
done