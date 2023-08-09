for num_samples in 10
do
for train_num_steps in 10 15
do
for sampling_batch_size in 16
do
for diffusion_time_steps in 300 2000
do
CUDA_VISIBLE_DEVICES=1 /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
--num_samples ${num_samples} \
--train_num_steps ${train_num_steps} \
--sampling_batch_size ${sampling_batch_size} \
--diffusion_time_steps ${diffusion_time_steps} \
--save_seq
done 
done
done
done

