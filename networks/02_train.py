import hydra 
import os
import numpy as np
import pickle
from utils.init import init_wandb, set_seed, open_log
from utils.data import create_dataloader
from utils.net import create_net
from utils.runner import train


@hydra.main(config_path="./config/train", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="prospective")
    set_seed(cfg.seed)
    open_log(cfg)

    # with open(cfg.data.path, 'rb') as fp:
    #     data = pickle.load(fp)

    allerrs_t = []
    for t in range(cfg.tstart, cfg.tend, cfg.tskip):
        all_errs = []
        for seed in range(cfg.numseeds):

            loaders = create_dataloader(cfg, t, seed)
            net = create_net(cfg)
            errs = train(cfg, net, loaders)
            all_errs.append(errs)
            print("Time %d, Seed %d, Error: %.4f" % (t, seed, np.mean(errs)))
        all_errs = np.array(all_errs)
        allerrs_t.append((t, all_errs))

    fdir = os.path.join('checkpoints', cfg.tag)
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, cfg.name + "_errs.pkl"), 'wb') as fp:
        pickle.dump(allerrs_t, fp)


if __name__ == "__main__":
    main()
