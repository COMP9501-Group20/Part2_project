# Installation and first run

The simulator target is **Ubuntu 20.04/22.04, Python 3.8 and Isaac Gym Preview 4**, with an NVIDIA GPU. Isaac Gym is installed separately from NVIDIA; it is not included in this repository. This is a legacy environment, not Isaac Lab or Isaac Sim. The full Linux/GPU stack has not been tested in this delivery.

## Environment

```bash
conda create -n comp9501 python=3.8 -y
conda activate comp9501
# A legacy CUDA 11.3 PyTorch build from the official previous-versions index.
python -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Obtain Isaac Gym Preview 4 from NVIDIA, then install its Python package.
python -m pip install -e /ABSOLUTE/PATH/TO/isaacgym/python

# From COM9501_Group20_Project:
python -m pip install -r requirements.txt
python -m pip install --no-deps -e rsl_rl
python -m pip install --no-deps -e legged_gym
python -c "import isaacgym; import torch; print(torch.__version__, torch.cuda.is_available())"
```

The torch command follows [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/). The remaining dependencies target the legacy Isaac Gym environment. Pinning these versions is a reproducibility starting point, not a tested installation guarantee for every host.

If `libpython3.8.so.1.0` is missing, activate the environment and run:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

## First checks

```bash
python -m unittest discover -s tests -v
python scripts/run.py train --variant full --dry-run
python scripts/run.py train --variant full --num-envs 32 --iterations 2 --headless
```

The last command is the first **physics integration** check. A successful CPU unit test is not a substitute. Check the new `runs/` folder for `run.json`, a checkpoint and TensorBoard events before launching a long experiment.

For a 4096-environment run, GPU memory demand may exceed a laptop GPU's capacity. Start small and keep environment count and rollout budget equal across compared variants. A reduced configuration is a different experiment and should be recorded as such.

## Local tests without Isaac Gym

The unit tests require PyTorch and NumPy only; they run on CPU and add the vendored algorithm package to the import path. The dry-run and sweep-plan commands use only the Python standard library. Plotting requires Matplotlib. No global package installation is needed just to inspect the configuration.

## Playback

`python scripts/run.py play --variant full --checkpoint /PATH/TO/model.pt` opens the Isaac Gym viewer for 20 seconds by default. Use `--horizon 60` for a longer clip. Record that viewer using your normal screen recorder, then store the checkpoint hash, seed and command alongside the recording. This records the behavior of the selected checkpoint for later comparison.
