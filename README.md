# Evaluating the effect of adding Gradient noise on the learning performance of deep neural networks.

The aim of the project is to explain and reproduce the result of Adding Gradient Noise Improves Learning for Very Deep Networks ([original paper](https://arxiv.org/abs/1511.06807)).

This was done as a team project. I was mainly in charge of empirical evaluation.

## getting started

```bash
# to run all the experiments
bash run.sh

# to run a specific experiment
python3 main.py --batch_size 1 --grad_noise --eta 0.01 --grad_clip --grad_clip_value 10.0 --init_weight_type "good" --exp_id "exp5_sgd_noise" --lr 0.1
```


## The jupyter notebook tutorial
[link](/evaluation_gaussian_noise_neural_network_tutorial.ipynb)
