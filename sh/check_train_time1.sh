for num_samples in 5 10 30
do
for train_num_steps in 100 150
do
CUDA_VISIBLE_DEVICES=1 /mlainas/teang1995/anaconda3/envs/PPG/bin/python main.py \
--num_samples ${num_samples} \
--train_num_steps ${train_num_steps}
done 
done