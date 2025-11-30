# DeepMIR-Fall-2025 Final Project team_123: Rap2Beat
The goal of this project is to builda Rap2Beat model that can be conditioned on a rap vocal track, and generate a new corresponding drum accompaniment which is rhythmically aligned and consistent with the vocal’s flow and genre.

## Install
Clone this repository and install the requirements (haven't build the requirements.txt, for now it's just a placeholder) with
```
git clone https://github.com/WeiFu3024/DeepMIR-2025-Fall-Final-Project-team_123.git
cd DeepMIR-2025-Fall-Final-Project-team_123
pip install requirement.txt
```
For teammate that using the lab server, you can run 
```
bash setup_link.sh
``` 
to setup the symbolic link to the preprocessed dataset and allin1 results.

## Dataset: RapBank
Clone the repository
```
git clone https://github.com/NZqian/RapBank
```

FILL IN THE DESCRIPTION OF HOW TO USE THE DATASET HERE.

## TODO
We should build a stable-audio-open finetuning pipeline in this repository later.