# SDCGDCP

## Welcome to SDCGDCP
**SDCGDCP (Synergistic Drug Combination Prediction via Graphormer and Drug-Cell Line Pair Graph)** <p align="justify">  Drug combination synergy is crucial in pharmacology, as it can enhance disease treatment efficacy or reduce drug resistance when administered in combination. Accurate prediction of drug combination synergy is vital for optimizing therapeutic regimens and improving treatment effectiveness. Existing computational methods primarily rely on drug sequence and structural features, making them difficult to capture complex network relationships and global information—especially lacking the ability to perform cross-modal fusion. In this study, we proposed a method called SDCGDCP for predicting drug combination synergy. It processed drug molecular structures (graph structures and molecular fingerprints), target biological activity information and integrated cell line whole-genome expression profiles to construct multi-level combined node representations. A drug-cell line pair graph was accordingly generated. SDCGDCP updated node and edge representations via a GatedGCN module and derived five types of structural encodings (centrality, spatial, edge, Laplacian positional and node-similarity encodings), after which the graph language model Graphormer is employed to capture long-range node interactions. Finally, drug combination synergy was predicted using an MLP and SoftMax classifier. Extensive evaluations showed that SDCGDCP outperforms other state-of-the-art methods on the DrugCombDB dataset, achieving an AUROC of 0.923 and AUPRC of 0.885. Ablation experiments validated the effectiveness of each feature and encoding module. Meanwhile, we conducted case analyses on the predicted drug combination synergies. The results were supported by evidence from several pharmaceutical studies. This highlights potential of SDCGDCP in enhancing drug synergy prediction and optimizing combination therapies.</p>

The flow chart of SDCGDCP is as follows:

![示例图片](./框架图.png)

## Directory Structure

```markdown
├── Datasets
│   ├── features
│   │   ├── Gene.csv           
│   │   ├── Target.csv            
│   │   ├── ECFP.csv            
│   │   └── Graph.csv             
│   └── samples					  
│       └── samples.csv            
├── config.py                     
├── data_preprocess.py                       
├── functional.py                            
├── layers.py                      
├── main.py                      
├── model.py                      
├── train_eval.py                      
└── utils.py                     
```

## Installation and Requirements

SDCGDCP has been tested in a Python 3.9 environment. It is recommended to use the same library versions as specified. Using a conda virtual environment is also recommended to avoid affecting other parts of the system. Please follow the steps below.

Key libraries and versions:

```markdown
├── torch              1.12.0+cu113
├── torch-geometric    2.5.3
├── torch-scatter      2.1.1
├── torch-sparse       0.6.18
├── networkx           3.2.1
├── scikit-learn       1.4.2
├── pandas             1.2.4
├── matplotlib         3.4.3
└── numpy              1.26.4        
```

### Step 1: Download Code and Data

Use the following command to download this project or download the zip file from the "Code" section at the top right:

```bash
git init 
git clone https://github.com/Philosopher-Zhao/SDCGDCP.git
```

### Step 2: Run the Model

Run the main script in the virtual environment:

```bash
python main.py
```

All results of the operation will be saved in the current directory.

## Citation 

If you use our tool and code, please cite our article and mark the project to show your support，thank you!

Citation format: 

Yuchen Zhang, Bingzhe Zhao, Zhuoqun Fu, Yiming Han, Beidan Liu and Xiujuan Lei, "Synergistic Drug Combination Prediction via Graphormer and Drug-Cell Line Pair Graph," 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Wuhan, China, 2025.

Paper Link:
