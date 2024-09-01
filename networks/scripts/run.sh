#python3 02_train.py  net.type=mlp name='mlp_ft_p20' tag=scenario2 numseeds=1 tstart=20 tend=2000 tskip=1 train.epochs=20 fine_tune=10 data.bs=8
#python3 02_train.py  net.type=mlp name='mlp_ft1_p20' tag=scenario2 numseeds=1 tstart=20 tend=2000 tskip=1 train.epochs=1 fine_tune=5 data.bs=8

#python3 02_train.py  net.type=mlp name='mlp_bgd_p20' tag=scenario2 numseeds=1 tstart=20 tend=1000 tskip=1 train.epochs=1 fine_tune=20 data.bs=8 bgd=True

# Scenario 3
# python3 02_train.py  net.type=mlp3 name='mlp_l' tag=scenario3 numseeds=1 tstart=50 tend=4000 tskip=200 train.epochs=100 data.bs=32 data.path='./data/synthetic/scenario3_period20.pkl'

# python3 02_train.py  net.type=prospective_mlp3 name='prospective_mlp_l' tag=scenario3 numseeds=1 tstart=50 tend=4000 tskip=200 train.epochs=100 data.bs=32 data.path='./data/synthetic/scenario3_period20.pkl'

python3 02_train.py  net.type=prospective_mlp_mnist name='prospective_mlp_l' tag=mnist_s2_v2 numseeds=1 tstart=500 tend=20001 tskip=1000 train.epochs=100 data.bs=32 data.path='./data/mnist/scenario2.pkl'


# python3 02_train.py  net.type=mlp_mnist name='mlp_l' tag=mnist_s2 numseeds=1 tstart=500 tend=20001 tskip=1000 train.epochs=10 data.bs=32 data.path='./data/mnist/scenario2.pkl'

# Scenario 3
# python3 02_train.py  net.type=mlp_mnist name='mlp_ft1' tag=mnist_s2 numseeds=1 tstart=500 tend=20001 tskip=500 train.epochs=1 fine_tune=True data.bs=8 data.path='./data/mnist/scenario2.pkl'

# python3 02_train.py  net.type=mlp_mnist name='mlp_bgd' tag=mnist_s2 numseeds=1 tstart=500 tend=20001 tskip=500 train.epochs=1 fine_tune=True data.bs=8 bgd=True data.path='./data/mnist/scenario2.pkl'
