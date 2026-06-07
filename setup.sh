#!/bin/bash

# Update the system
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip git

# Create virtural environment
sudo pip3 install virtualenv
apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.7-dev python3.7-venv
mkdir virtual_env
/usr/bin/python3.7 -m venv ~/virtual_env/venv_with_python3.7
source ~/virtual_env/venv_with_python3.7/bin/activate
pip install -r requirements.txt


