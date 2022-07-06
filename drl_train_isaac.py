import argparse
import os.path as osp

from hydra import compose, initialize
from isaacgymenvs import train  # this registers necessary resolvers + imports IsaacSim
from envs import isaacgym_task_map
from omegaconf import OmegaConf

from drl.utils.common import construct_config
from drl.utils.registry import drl_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("isaac_cfg", help="path to config.yaml in IsaacGymEnvs")
    parser.add_argument("drl_overrides", nargs="*", help="YACS overrides for drl repo")
    parser.add_argument(
        "-o",
        "--isaac-overrides",
        nargs="*",
        help="HYDRA overrides for IsaacGymEnvs repo",
    )
    parser.add_argument(
        "-c",
        "--drl-config",
        help="path to drl config yaml (uses default.yaml by default)",
    )
    args = parser.parse_args()

    config_name = osp.splitext(osp.basename(args.isaac_cfg))[0]
    config_path = osp.dirname(args.isaac_cfg)
    initialize(config_path=config_path, job_name="drl_isaac_training")
    cfg = compose(config_name=config_name, overrides=args.isaac_overrides)

    drl_config = construct_config(args.drl_config, args.drl_overrides)
    cfg.seed = drl_config.SEED  # sync seed with drl's cfg

    print(OmegaConf.to_yaml(cfg))
    print(drl_config)
    env = isaacgym_task_map[cfg.task_name](
        cfg=OmegaConf.to_object(cfg.task),
        sim_device="cuda:0",
        graphics_device_id="cuda:0",
        headless=cfg.headless,
    )
    runner_cls = drl_registry.get_runner(drl_config.RUNNER.name)
    runner = runner_cls(drl_config, env)
    runner.train()


if __name__ == "__main__":
    main()
