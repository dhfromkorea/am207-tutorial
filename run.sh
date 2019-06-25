# Experiment 3

# SGD 
#python3 main.py --batch_size 1 --init_weight_type "simple" --exp_id "exp3_sgd" --grad_clip --grad_clip_value 10.0 --lr 0.01 --simple_init_std 1.0 --weight_decay 0.01 --n_epoch 20

# SGD w/ noise
#python3 main.py --batch_size 1 --grad_noise --eta 0.01 --gamma 0.55 --init_weight_type "simple" --exp_id "exp3_sgd_noise" --grad_clip --grad_clip_value 10.0 --lr 0.01 --simple_init_std 1.0 --weight_decay 0.01 --n_epoch 20


# experiment 5

# SGD 
python3 main.py --batch_size 1 --grad_clip --grad_clip_value 10.0 --init_weight_type "good" --exp_id "exp5_sgd" --lr 0.1 --cuda

# SGD w/ noise
python3 main.py --batch_size 1 --grad_noise --eta 0.01 --grad_clip --grad_clip_value 10.0 --init_weight_type "good" --exp_id "exp5_sgd_noise" --lr 0.1 --cuda


# Experiment 6

# SGD 
#python3 main.py --batch_size 1 --grad_clip --grad_clip_value 10.0 --init_weight_type "bad" --exp_id "exp6_sgd" --lr 0.1 --weight_decay 0.001 --n_epoch 20

# SGD w/ noise
#python3 main.py --batch_size 1 --grad_noise --eta 1.0 --grad_clip --grad_clip_value 10.0 --init_weight_type "bad" --exp_id "exp6_sgd_noise" --lr 0.1 --weight_decay 0.001 --n_epoch 20


