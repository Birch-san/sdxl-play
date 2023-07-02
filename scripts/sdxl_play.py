from os.path import join, dirname
import sys
from pathlib import Path
repo_root: Path = Path(__file__).parents[1]
sys.path.append(str(repo_root / 'src'))
sys.path.append(str(repo_root / 'lib/generative-models'))

from omegaconf import OmegaConf, DictConfig

out_dir='out'
out_name='img.png'
out_path=join(out_dir, out_name)

h=1024
w=1024

# TODO: work out a module-relative way to load this package-owned resource
config: DictConfig = OmegaConf.load('configs/inference/sd_xl_base.yaml')