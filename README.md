# NEOSoVIT
## Fish diffusion reimplementation for singing voice cloning

### Step 0: Create new **Conda** environment
```
conda create -n neosovit python=3.10
conda activate neosovit
```
### Step 1: Install requirements
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch-lightning

pip install -U openmim
mim install mmengine

pip install librosa
pip install loguru
pip install wandb
pip install pyloudnorm
pip install transformers
pip install torchcrepe
pip install praat-parselmouth
pip install pyworld
pip install ffmpeg
```

### Step 2: Download pretrained model
https://github.com/fishaudio/fish-diffusion/releases

(You are looking for the **ckpt** and **finetune.py**)