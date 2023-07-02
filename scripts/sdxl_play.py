from os.path import join
import sys
from pathlib import Path
repo_root: Path = Path(__file__).parents[1]
sys.path.append(str(repo_root / 'src'))
sys.path.append(str(repo_root / 'lib/generative-models'))

from omegaconf import OmegaConf, DictConfig
from importlib.util import find_spec
from sgm.models.diffusion import DiffusionEngine
from sgm.util import instantiate_from_config

out_dir='out'
out_name='img.png'
out_path=join(out_dir, out_name)

h=1024
w=1024

gen_models_path = Path(find_spec('sgm').origin).parents[1]
base_config_path: Path = gen_models_path / 'configs/inference/sd_xl_base.yaml'

config: DictConfig = OmegaConf.load(base_config_path)
# model: DiffusionEngine = instantiate_from_config(config.model)
model = DiffusionEngine(**config.model.params)

pass