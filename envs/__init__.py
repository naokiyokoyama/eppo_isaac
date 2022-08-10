from isaacgymenvs.tasks import isaacgym_task_map

from .ant import AntV2
from .humanoid import HumanoidV2
from .anymal_terrain import AnymalTerrainV2

isaacgym_task_map[AntV2.orig_name] = AntV2
isaacgym_task_map[HumanoidV2.orig_name] = HumanoidV2
isaacgym_task_map[AnymalTerrainV2.orig_name] = AnymalTerrainV2
