conda create -n isaac_dm -y python=3.8 &&
conda install -n isaac_dm pytorch cudatoolkit=11.6 gdown -c pytorch -c conda-forge -y &&
mkdir isaacdm &&
cd isaacdm &&
git clone https://github.com/naokiyokoyama/IsaacGymEnvs -b use_as_library &&
git clone https://github.com/naokiyokoyama/drl -b refactor_1 &&
git clone https://github.com/naokiyokoyama/eppo_isaac &&
~/.conda/envs/isaac_dm/bin/gdown --fuzzy https://drive.google.com/file/d/1A4DwFb8Jj-PCFcKZE_SJA4b_oN5t9XSm/view?usp=sharing &&
tar -xvf IsaacGym_Preview_3_Package.tar.gz &&
rm IsaacGym_Preview_3_Package.tar.gz &&
~/.conda/envs/isaac_dm/bin/pip install -e IsaacGymEnvs/ &&
~/.conda/envs/isaac_dm/bin/pip install -e drl/ &&
~/.conda/envs/isaac_dm/bin/pip install -r drl/requirements.txt &&
~/.conda/envs/isaac_dm/bin/pip install -e isaacgym/python &&
~/.conda/envs/isaac_dm/bin/pip install -r eppo_isaac/requirements.txt &&
cd &&
conda init && source ~/.bashrc &&
git clone https://github.com/naokiyokoyama/my_env &&
python my_env/aliases/add_aliases.py -s n &&
python my_env/aliases/generate*
