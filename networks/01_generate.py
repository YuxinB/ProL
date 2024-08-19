import hydra 
from utils.init import init_wandb, set_seed, open_log
from utils.data import SyntheticScenario2,SyntheticScenario3
from utils.data import MNISTScenario2, CIFARScenario2


@hydra.main(config_path="./config/gen", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="prospective")
    set_seed(cfg.seed)
    open_log(cfg)

    if cfg.scenario == 2:
        if cfg.data == 'synthetic':
            datagen = SyntheticScenario2(cfg)
        elif cfg.data == 'mnist':
            datagen = MNISTScenario2(cfg)
        elif cfg.data == 'cifar':
            datagen = CIFARScenario2(cfg)

    elif cfg.scenario == 3:
        datagen = SyntheticScenario3(cfg)

    datagen.generate_data()
    datagen.store_data()


if __name__ == "__main__":
    main()
