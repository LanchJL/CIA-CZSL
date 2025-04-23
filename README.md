# Learning Clique-based Inter-class Affinity for Compositional Zero-shot Learning
* **Title**: **Learning Clique-based Inter-class Affinity for Compositional Zero-shot Learning**
* **Institutes**: Nanjing University of Science and Technology, Nanjing Forestry University, Newcastle University

## Environment Setup
```bash
conda create --name cia python=3.8
conda activate cia
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install git+https://github.com/openai/CLIP.git
```

The remaining dependencies can be installed using `pip install -r requirements.txt`, with the list of required packages found in the `./requirements.txt` file.

Additionally, the CLIP model weights can be downloaded from [CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt). After downloading, please place the weights in the `./clip_modules/` directory.

## Datasets
The dataset splits and their respective attributes are defined in the `./utils/download_data.sh` script. For the full installation instructions, please refer to [CGE&CompCos](https://github.com/ExplainableML/czsl).  
To download the datasets, you can run:
```shell
bash ./utils/download_data.sh
```

## Training
If you would like to train the model from scratch, for example, on the **UT-Zappos** dataset, use the following command:
```shell
python -u train.py \
--clip_arch ./clip_modules/ViT-L-14.pt \
--dataset_path <path_to_UT-Zap50k> \
--save_path <path_to_logs> \
--yml_path ./config/ut-zappos.yml \
--num_workers 4 \
--seed 0 \
--adapter
```

For training on **OW-CZSL**, use this command:
```shell
python -u train.py \
--clip_arch ./clip_modules/ViT-L-14.pt \
--dataset_path <path_to_UT-Zap50k> \
--save_path <path_to_logs> \
--yml_path ./config/ut-zappos-ow.yml \
--num_workers 4 \
--seed 0 \
--adapter
```

## Acknowledgements
The code we have released is built upon excellent repositories that have greatly contributed to our work:
* [Troika](https://github.com/bighuang624/Troika?tab=readme-ov-file)
