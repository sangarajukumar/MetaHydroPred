<h1 align="center">MetaHydroPred</h1>
<p align="center"><a href="https://balalab-skku.org/MetaHydroPred/">🌐 Webserver (CBBL-SKKU)</a></p>

The official implementation of **MetaHydroPred: Accurate and Interpretable Prediction of Hydrogen Production and Current Density in Microbial Electrolysis Cells Using a Meta-learning Framework**

## Installing environment
First, install [Miniconda](https://docs.anaconda.com/miniconda/) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).  
Then create and activate the environment:

```bash
conda create -n metahydropred python=3.7.16
conda activate metahydropred
```
Next, install the required dependencies:
```bash
cd MetaHydroPred/
python -m pip install -r requirements.txt --no-cache-dir
```

## Input Data Format

Your input CSV file should contain the appropriate features for each substrate type.

**Note:** Input data must follow the exact column format specified below. Example training and testing datasets are available in `dataset/benchmark-dataset/` for reference.

### H₂ Production Rate

**Example file (`h2_acetate_test.csv`):**
```csv
Substrate concentration,Reactor working volume,Cathode projected surface area,S/V ratio,Temperature,Applied voltage,H2 production rate
1.0,95,30,32,33,0.9,1.09
1.4,34,12,35,30,0.9,1.1
```
### Current Density

**Example file (`cd_acetate_test.csv`):**
```csv
Substrate concentration,Reactor working volume,Cathode projected surface area,S/V ratio,Temperature,Applied voltage,Current density
1.1,50,50,100,30,1.0,360
1.6,28,7,25,30,0.9,85
```

## Usage

### H₂ Production Rate Prediction
```bash
python H2Production_predictor.py --type  --input  --output 
```

**Substrate types:** `all-organic`, `acetate`, `complex-substrate`

**Example:**
```bash
python H2Production_predictor.py --type complex-substrate --input test_data.csv --output h2_complex-substrate_predictions.csv
```

### Current Density Prediction
```bash
python CurrentDensity_predictor.py --type  --input  --output 
```

**Substrate types:** `all-organic`, `acetate`, `complex-substrate`

**Example:**
```bash
python CurrentDensity_predictor.py --type acetate --input test_data.csv --output cd_acetate_predictions.csv
```
## Citation

If you use MetaHydroPred in your research, please cite:

```bash
@article{,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```