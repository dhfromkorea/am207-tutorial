# Experimetn 1

# SGD 
python3 main_torch.py --batch_size 1 --init_weight_type "simple" --exp_id "exp1_sgd"

# SGD w/ noise
python3 main_torch.py --batch_size 1 --grad_noise --eta 0.01 --gamma 0.55 --init_weight_type "simple" --exp_id "exp1_sgd_noise"


# experiment 5

# SGD 
python3 main_torch.py --batch_size 1 --grad_clip --grad_clip_value 10.0 --init_weight_type "good" --exp_id "exp5_sgd"

# SGD w/ noise
python3 main_torch.py --batch_size 1 --grad_noise --eta 0.01 --grad_clip --grad_clip_value 10.0 --init_weight_type "good" --exp_id "exp5_sgd_noise"


# Experimetn 6

# SGD 
python3 main_torch.py --batch_size 1 --grad_clip --grad_clip_value 10.0 --init_weight_type "bad" --exp_id "exp6_sgd"

# SGD w/ noise
python3 main_torch.py --batch_size 1 --grad_noise --eta 0.01 --grad_clip --grad_clip_value 10.0 --init_weight_type "bad" --exp_id "exp6_sgd_noise"

