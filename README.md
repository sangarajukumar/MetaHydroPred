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

## Input Data Format

Your input CSV file should contain the appropriate features for each substrate type.

**Note:** Ensure your test data matches the format of the training datasets provided in `dataset/`.
