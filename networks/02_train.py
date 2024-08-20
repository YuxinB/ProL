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

    allerrs_t = []
    for seed in range(cfg.numseeds):
        print("Seed %d =================" % seed)
        all_errs = []
        net = create_net(cfg)

        for t in range(cfg.tstart, cfg.tend, cfg.tskip):
            # Create new network if not fine-tuning
            loaders = create_dataloader(cfg, t, seed)
            if (cfg.fine_tune is None) and (cfg.bgd is None):
                net = create_net(cfg)

            errs = train(cfg, net, loaders)
            all_errs.append((t, errs))
            print("Time %d, Seed %d, Error: %.4f" % (t, seed, np.mean(errs)))

        allerrs_t.append(all_errs)

    fdir = os.path.join('checkpoints', cfg.tag)
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, cfg.name + "_errs.pkl"), 'wb') as fp:
        pickle.dump(allerrs_t, fp)


if __name__ == "__main__":
    main()
