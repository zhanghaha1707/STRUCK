# STRUCK

code for "Structural Attack for Models of Code Representation"

- data: store the relevant data required by the code
- dataset: Related classes that deal with code data
- model: code representation models
- ModelHandler: model training and baseline methods of adversrial attack
- struck: code for STRUCK

# Process
## Step0: prepare data
**0-1: split dataset (data folder)**

- `python read_data.py`: divided and stored in the corresponding train/dev/test file according to the proportion of 16:4:5

**0-2: build vocabulary for models (dataset folder)**

- `python vocabClass.py --name lstm ` 

**0-3: Processing dataset (dataset folder)**

- `Python dataClass.py --name lstm` (dataClass support [lstm,gru,codebert,graphcodebert])
- If processing graph model data: `python GraphDataClass.py`

## Step1: training models

> (ModelHandler folder)

- `python trainer.py --train_model lstm --train True --val True --test True`

## Step2: Attack models

### Baselines 
> (ModelHandler folder)

#### For all modesls: 
> attack methods: MHM, random_rename($\text{I-CARROT}_{\text{A}}\text{-S}$$), insert($$\text{S-CARROT}_\text{A}$)

- `python attacker_baseline.py --attack_model lstm --attack_way mhm --attack_data test `

#### For pre-trained models: 
> attack methods: ALERT

- `python retrain_attacker.py --arrack_model codebert`

### STRUCK (struck folder)

- `python attacker.py --attack_model lstm --attack_data test`

## Step3: Attack enhance

### Generate Enhance dataset by STRUCK 
> (struck folder)

- `python attacker.py --attack_model lstm --attack_data train`

### Re-traing models 
> (ModelHandler folder)

- `python trainer.py --train_model lstm --is_enhance True --enhance_size 15000 --train True --val True --test True`