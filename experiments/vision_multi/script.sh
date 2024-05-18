# #### MNIST ####
# "lr": 1e-3,         
# "batchsize": 64,
# "epochs": 500,
# "contextlength": 200,

# lower bound baseline
python run_baseline.py -m t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="cnn" device='cuda:3' hydra.launcher.n_jobs=10

# for others
python run_vision_multi.py -m t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="cnn" device='cuda:0' hydra.launcher.n_jobs=8
python run_vision_multi.py -m t=0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="timecnn" device='cuda:0' hydra.launcher.n_jobs=8

# for proformer
CUDA_VISIBLE_DEVICES=1 python run_vision_multi.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="proformer" device='cuda:0' hydra.launcher.n_jobs=10




# #### CIFAR-10 ####
# "lr": 1e-3,         
# "batchsize": 32,
# "epochs": 500,
# "contextlength": 80,

# # lower bound baseline
# python run_baseline.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="resnet" device='cuda:1' hydra.launcher.n_jobs=10

# # for others
# python run_vision_multi.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="resnet" device='cuda:0' hydra.launcher.n_jobs=10
# python run_vision_multi.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="timeresnet" device='cuda:1' hydra.launcher.n_jobs=10

# # for conv_proformer
# python run_vision_multi.py -m t=0,100,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000 method="conv_proformer" device='cuda:0' hydra.launcher.n_jobs=4

CUDA_VISIBLE_DEVICES=1 python run_vision_multi.py -m t=1000 method="conv_proformer" device='cuda:0' hydra.launcher.n_jobs=1