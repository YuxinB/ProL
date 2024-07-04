# MLP
# python run_synthetic_markov.py -m device='cuda:0' method="mlp" epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# time MLP
# python run_synthetic_markov.py -m device='cuda:1' method="timemlp" epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# prosp. transformer
# python run_synthetic_markov.py -m device='cuda:1' method="proformer" multihop=True epochs=150 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=5

# transformer
# python run_synthetic_markov.py -m device='cuda:3' method="proformer" multihop=False epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# Baseline 1 (Oracle)
# python run_baseline_1.py -m device='cuda:0' method="mlp" epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

python run_synthetic_markov.py -m device='cuda:0' eps=0.0001 method="proformer" multihop=False epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8
python run_synthetic_markov.py -m device='cuda:1' eps=0.001 method="proformer" multihop=False epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8
python run_synthetic_markov.py -m device='cuda:2' eps=0.01 method="proformer" multihop=False epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8
python run_synthetic_markov.py -m device='cuda:3' eps=0.1 method="proformer" multihop=False epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8

# python run_synthetic_markov.py -m device='cuda:0' eps=0.001 method="timemlp" epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8
# python run_synthetic_markov.py -m device='cuda:1' eps=0.01 method="timemlp" epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8
# python run_synthetic_markov.py -m device='cuda:2' eps=0.1 method="timemlp" epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8
# python run_synthetic_markov.py -m device='cuda:3' eps=0.2 method="timemlp" epochs=300 batchsize=64 t=0,200,500,700,1000,1200,1500,1700,2000,2500 hydra.launcher.n_jobs=8