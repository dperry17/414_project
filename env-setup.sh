#!/bin/bash

sudo apt update
sudo apt install -y direnv python3 python3-pip python3-venv

if [ ! -d ".pyenv" ]; then
    python3 -m venv .pyenv
    source .pyenv/bin/activate
    pip3 install -r requirements.txt
fi

# Hook direnv into bash
if [ -z "$(cat ~/.bashrc | grep 'eval \"$(direnv hook bash)\"')" ]; then
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
fi

source ~/.bashrc
direnv allow .
