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