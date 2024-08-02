#$python3 02_train.py  net.type=mlp name='mlp_ft_p20' tag=scenario2 numseeds=1 tstart=20 tend=2000 tskip=1 train.epochs=20 fine_tune=10 data.bs=8
#python3 02_train.py  net.type=mlp name='mlp_ft1_p20' tag=scenario2 numseeds=1 tstart=20 tend=2000 tskip=1 train.epochs=1 fine_tune=5 data.bs=8

#python3 02_train.py  net.type=mlp name='mlp_bgd_p20' tag=scenario2 numseeds=1 tstart=20 tend=1000 tskip=1 train.epochs=1 fine_tune=20 data.bs=8 bgd=True

# Scenario 3
python3 02_train.py  net.type=mlp3 name='mlp_l' tag=scenario3 numseeds=1 tstart=50 tend=4000 tskip=200 train.epochs=100 data.bs=32 data.path='./data/synthetic/scenario3_period20.pkl'

python3 02_train.py  net.type=prospective_mlp3 name='prospective_mlp_l' tag=scenario3 numseeds=1 tstart=50 tend=4000 tskip=200 train.epochs=100 data.bs=32 data.path='./data/synthetic/scenario3_period20.pkl'
