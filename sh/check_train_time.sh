for num_samples in 1
do
for train_num_steps in 1
do
CUDA_VISIBLE_DEVICES=0 /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
--num_samples ${num_samples} \
--train_num_steps ${train_num_steps}
done 
done