import hydra 
from utils.init import init_wandb, set_seed, open_log
from utils.data import create_dataloader
from utils.net import create_net
from utils.runner import train


@hydra.main(config_path="./config/train", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="prospective")
    set_seed(cfg.seed)
    open_log(cfg)


    for seed in range(1):
        for t in range(500, 501, 10):

            loaders = create_dataloader(cfg, t, seed)
            net = create_net(cfg)
            train(cfg, net, loaders)

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
