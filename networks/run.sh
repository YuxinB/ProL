#$python3 02_train.py  net.type=mlp name='mlp_ft_p20' tag=scenario2 numseeds=1 tstart=20 tend=2000 tskip=1 train.epochs=20 fine_tune=10 data.bs=8
#python3 02_train.py  net.type=mlp name='mlp_ft1_p20' tag=scenario2 numseeds=1 tstart=20 tend=2000 tskip=1 train.epochs=1 fine_tune=5 data.bs=8

python3 02_train.py  net.type=mlp name='mlp_bgd_p20' tag=scenario2 numseeds=1 tstart=20 tend=1000 tskip=1 train.epochs=1 fine_tune=20 data.bs=8 bgd=True


