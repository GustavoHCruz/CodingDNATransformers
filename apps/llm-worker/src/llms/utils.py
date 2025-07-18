import os
import random

import numpy as np
import torch


def set_seed(seed) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(True)

  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"