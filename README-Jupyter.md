## Install Ollama on WSL
```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama --version
    ollama version is 0.7.0

ollama run llama3.2:1b
>>> /bye

curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

node -v
    v20.19.2
npm -v
    10.8.2

npm prefix -g
    /usr

sudo apt update
sudo apt install ffmpeg -y
ffmpeg -version
    ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
    ...

echo 'export BROWSER="/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"' >> ~/.bashrc
source ~/.bashrc
```

## Create a virtual environment
```bash
python3 -m venv wsl_venv
source wsl_venv/bin/activate
python3 -m pip install --upgrade pip

chmod +x ./requirements.sh
./requirements.sh

git config --global credential.helper store
```

### Start up Jupyter Lab
```cmd
source wsl_venv/bin/activate
jupyter lab
jupyter lab --ThemeManager.theme="JupyterLab Dark"

```
