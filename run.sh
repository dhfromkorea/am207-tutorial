# Experimetn 1

# SGD 
python3 main_torch.py --batch_size 1 --init_weight_type "simple"

# SGD w/ noise
python3 main_torch.py --batch_size 1 --grad_noise --eta 0.01 --gamma 0.55 --init_weight_type "simple"


# experiment 5

# SGD 
python3 main_torch.py --batch_size 1 --grad_clip --grad_clip_value 10.0 --init_weight_type "good"

# SGD w/ noise
python3 main_torch.py --batch_size 1 --grad_noise --eta 0.01 --grad_clip --grad_clip_value 10.0 --init_weight_type "good"


# Experimetn 6

# SGD 
python3 main_torch.py --batch_size 1 --grad_clip --grad_clip_value 10.0 --init_weight_type "bad"

# SGD w/ noise
python3 main_torch.py --batch_size 1 --grad_noise --eta 0.01 --grad_clip --grad_clip_value 10.0 --init_weight_type "bad"

