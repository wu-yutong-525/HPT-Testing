import os
import numpy as np
import hydra
import torch
from hpt.utils import utils, model_utils
from omegaconf import OmegaConf
from hpt.models.policy import Policy
import os
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler
from env.mujoco.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS
from collections import OrderedDict

from hydra import compose, initialize

import time
import collections
class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0])
        else:
            return 0.0
        
policy = Policy.from_pretrained("hf://liruiw/hpt-base")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
domain = "mujoco_metaworld"
with initialize(version_base="1.2", config_path="experiments/configs"):
    cfg = compose(config_name="config", overrides=[f"env={domain}"])

cfg.dataset.episode_cnt = 10 # modify
dataset = hydra.utils.instantiate(
        cfg.dataset, dataset_name=domain, env_rollout_fn=cfg.dataset_generator_func, **cfg.dataset
    )
normalizer = dataset.get_normalizer()

####### set up model
utils.update_network_dim(cfg, dataset, policy)
policy.init_domain_stem(domain, cfg.stem)
policy.init_domain_head(domain, normalizer, cfg.head)
policy.finalize_modules()
policy.print_model_stats()
policy.to(device)
print("policy action normalizer:", policy.normalizer[domain].params_dict["action"]["input_stats"].max)

 
####### train one iter
train_loader = data.DataLoader(dataset, **cfg.dataloader)
batch = next(iter(train_loader))
batch["data"] = utils.dict_apply(batch["data"], lambda x: x.to(device).float())
output = policy.compute_loss(batch)


####### run rollout
RESOLUTION = (128, 128)
camera_name="view_1"
env_name = "reach-v2"
def get_observation_dict(o, img):
    step_data = {"state": o, "image": img}
    return OrderedDict(step_data)

env = ALL_ENVS[env_name]()
env._partially_observable = False
env._freeze_rand_vec = False
env._set_task_called = True
img = env.sim.render(*RESOLUTION, mode="offscreen", camera_name=camera_name)[:, :, ::-1].copy()
o = env.reset()
step_data = get_observation_dict(o, img)
policy.reset()

for _ in range(env.max_path_length):
    a = policy.get_action(step_data)
    o, r, done, info = env.step(a)
    img = env.sim.render(*RESOLUTION, mode="offscreen", camera_name=camera_name)[:, :, ::-1]
    step_data = get_observation_dict(o, img)
    print(a)
    break

        
fps_measure = FPS()
for _ in range(50):
    output = policy.get_action(step_data)
    fps_measure()
print(f"FPS: {fps_measure():.3f}")