# DynWGraphsPaper
Code for analysis of paper about dynmical sparse weighted graphs 

# Enviroment Set Up
- If you need to install conda, do:
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh

- Create the env
    conda env create --file requirements_conda.yml 
    conda activate dynwgraphs
    pip install -e ./src/dynwgraphs/
    pip install -e ./src/proj_utils/


 - set project path in config file proj_config.yml

