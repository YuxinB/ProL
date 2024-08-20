# python3 01_generate.py seq_len=10000 num_seeds=10 data=synthetic scenario=2

# python3 02_train.py net.type=mlp name='mlp_erm' \
#     tag=scenario2 numseeds=5 \
#     tstart=20 tskip=200 tend=2001

# python3 02_train.py net.type=prospective_mlp name='mlp_prospective' \
#     tag=scenario2 numseeds=5 \
#     tstart=20 tskip=200 tend=2001 

# python3 02_train.py net.type=mlp name='mlp_ft1' \
#     tag=scenario2 numseeds=5 \
#     tstart=20 tskip=1 tend=2001  \
#     train.epochs=1 fine_tune=5 data.bs=8

# python3 02_train.py net.type=mlp name='mlp_bgd' \
#     tag=scenario2 numseeds=5 \
#     tstart=20 tskip=1 tend=2001 \
#     train.epochs=1 fine_tune=20 data.bs=8 bgd=True


# python3 01_generate.py seq_len=40000 num_seeds=5 data=mnist scenario=2


python3 02_train.py  net.type=mlp_mnist name='erm_mlp' \
    dev='cuda:0' tag=mnist_s2 numseeds=5 \
    tstart=20 tend=5021 tskip=250 \
    train.epochs=10 data.bs=32 data.path='./data/mnist/scenario2.pkl'

# python3 02_train.py  net.type=prospective_mlp_mnist name='prospective_mlp' \
#     dev='cuda:0' tag=mnist_s2 numseeds=5 \
#     tstart=20 tend=5021 tskip=250 \
#     train.epochs=100 data.bs=32 data.path='./data/mnist/scenario2.pkl'


# python3 02_train.py net.type=mlp_mnist name='mlp_ft1' \
#     dev='cuda:0' tag=mnist_s2 numseeds=5 \
#     tstart=20 tend=5001 tskip=1 \
#     train.epochs=1 fine_tune=16 data.bs=8 data.path='./data/mnist/scenario2.pkl'

# python3 02_train.py net.type=mlp_mnist name='mlp_bgd' \
#     dev='cuda:0' tag=mnist_s2 numseeds=5 \
#     tstart=20 tend=5001 tskip=1 \
#     train.epochs=1 fine_tune=16 data.bs=8 bgd=True data.path='./data/mnist/scenario2.pkl'
