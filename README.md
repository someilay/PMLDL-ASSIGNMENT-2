## PMLDL ASSIGNMENT 2
#### Ilia Milioshin, i.mileshin@innopolis.university, B20-RO-01

---

## Setup

* Clone repo: 
```bash
git clone https://github.com/someilay/PMLDL-ASSIGNMENT-2.git
```

* Enter to cloned repo
```bash
cd PMLDL-ASSIGNMENT-2
```

* Create a virtual environment

For Unix/macOS
```bash
python3 -m venv venv
```
For Windows
```bash
py -m venv venv
```
* Activating a virtual environment


For Unix/macOS
```bash
source venv/bin/activate
```
For Windows
```bash
.\venv\Scripts\activate
```
* Installing a packages
```bash
pip install -r requirements.txt
```
* Install [pytorch](https://pytorch.org/get-started/locally/#start-locally)
* Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

* Setup `PYTHONPATH`

For Unix/macOS
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

For Windows (PowerShell)
```bash
set PYTHONPATH=$env:PYTHONPATH;$PWD
```

* Download data
```bash
python src/get_data.py
```

* Generate intermediate representations
```bash
python src/get_intermediate.py
```

---

## Train model

In the notebooks, you can see how are model built and trained. However, for manual training I recommend to use `train.py` script:

```bash
python src/train.py
```

More custom options you can see if run:

```bash
python src/train.py -h
```

---

## Evaluation

To evaluate a trained model you should run:

```bash
python benchmark/evaluate.py
```

The model would be evaluated on [MovieLens](https://grouplens.org/datasets/movielens/100k/) train/test sets. There are two types of models with and without features support. To get more information:

```bash
python benchmark/evaluate.py -h
```

To recommend a movie use:
```bash
python benchmark/recommend_movie.py -ui <user id> -vd <viewed movies> -mt <model type>
```

Example of usage:
```bash
python benchmark/recommend_movie.py -ui 12 -vd 50 181 -mt with-feature
```

Output:
```
Importing stuff...
Loading the data
Loading the model

Recommendation id: 174
Recommendation title: Raiders of the Lost Ark (1981)
Recommendation mean rating: 4.252
```

For help:
```bash
python benchmark/recommend_movie.py -h
```

--- 

## Current metrics:

* Without features support:

Mean test recall `0.2982`, Mean test precision `0.2855`

* With features support:

Mean test recall `0.3224`, Mean test precision `0.3088`
