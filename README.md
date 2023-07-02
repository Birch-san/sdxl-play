# SDXL-Play

**Note: this is a work-in-progress. If you follow these instructions you will hit a dead end.**

This repository will try to provide instructions and a Python script for invoking [SDXL](https://stability.ai/blog/sdxl-09-stable-diffusion) txt2img.

SDXL useful links:

- [`generative-models` library](https://github.com/Stability-AI/generative-models)
- [VAE](https://huggingface.co/stabilityai/sdxl-vae)
- [base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
- [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9)

This repository assumes that you already have access to the base/refiner weights (e.g. you have been granted Researcher Early Access).

## Setup

All instructions are written assuming your command-line shell is bash, and your OS is Linux.

### Repository setup

Clone this repository (this will also retrieve the [`generative-models`](https://github.com/Stability-AI/generative-models) submodule):

```bash
git clone --recursive https://github.com/Birch-san/sdxl-play.git
cd sdxl-play
```

### Create + activate a new virtual environment

This is to avoid interfering with your current Python environment (other Python scripts on your computer might not appreciate it if you update a bunch of packages they were relying on).

Follow the instructions for virtualenv, or conda, or neither (if you don't care what happens to other Python scripts on your computer).

#### Using `venv`

**Create environment**:

```bash
python -m venv venv
pip install --upgrade pip
```

**Activate environment**:

```bash
. ./venv/bin/activate
```

**(First-time) update environment's `pip`**:

```bash
pip install --upgrade pip
```

#### Using `conda`

**Download [conda](https://www.anaconda.com/products/distribution).**

_Skip this step if you already have conda._

**Install conda**:

_Skip this step if you already have conda._

Assuming you're using a `bash` shell:

```bash
# Linux installs Anaconda via this shell script. Mac installs by running a .pkg installer.
bash Anaconda-latest-Linux-x86_64.sh
# this step probably works on both Linux and Mac.
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
conda init
```

**Create environment**:

```bash
conda create -n p311-sdxl python=3.11
```

**Activate environment**:

```bash
conda activate p311-sdxl
```

### Install package dependencies

**Ensure you have activated the environment you created above.**

(Optional) treat yourself to latest nightly of PyTorch, with support for Python 3.11 and CUDA 12.1:

```bash
# CUDA
pip install --upgrade --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu121
# Mac
pip install --upgrade --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Install dependencies:

```bash
pip install -r requirements_inference.txt
```

We deliberately avoid installing `generative-models`' requirements files, as lots of the dependencies there exist to support training or watermarking.

## Run:

From root of repository:

```bash
python -m scripts.sdxl_play
```

## License

3-clause [BSD](https://en.wikipedia.org/wiki/BSD_licenses) license; see [`LICENSE.txt`](LICENSE.txt)