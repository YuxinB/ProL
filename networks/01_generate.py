import hydra 
from utils.init import init_wandb, set_seed, open_log
from utils.data import SyntheticScenario2,SyntheticScenario3


@hydra.main(config_path="./config/gen", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="prospective")
    set_seed(cfg.seed)
    open_log(cfg)

    # datagen = SyntheticScenario2(cfg)
    datagen = SyntheticScenario3(cfg)
    datagen.generate_data()
    datagen.store_data()


if __name__ == "__main__":
    main()
