# Variational Autoencoder - main.py

Command line interface to variational autoencoder used for generating hamiltonian shapes

## Setup

The `boot.py` script is used to build the environment. It builds a python virtual environment and runs the necessary pip commands. We were not able to use a `requirement.txt` file solution since some of the dependencies need to be installed in a certain order.

```$ python boot.py```

If you would prefer to run the commands yourself, here are the necessary pip commands (in order):

```
pip install --upgrade pip
pip install torch==2.3.0 wheel setuptools
pip install torch-sparse torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install git+https://github.com/tomfu1/nablaDFT.git@hack/manifest
pip install scikit-learn numpy PyYAML boto3
```

## Data

TODO

## Options

The command line interface provides 4 options.

* `-c, --config CONFIG`
  
  Location of YAML configuration file. Defaults to `main.yaml`.

* `-g, --generate FILE`

  Loads model from `model_path` and generates a new hamiltonian shape. Outputs shape to `FILE` as tensor then exits.

* `-j, --json`

  Output final statistics in json format. 

* `-v, --verbose`

  Output batch statistics.

## Configuration

`main.py` uses a YAML file to configure parameters. We provide a `main.default.yaml` so that the user can easily see what the default values for each parameter are. If the user would like to edit some parameters, we recommend creating and editing a new `main.yaml` file. The user only needs to provide values you intend on changing from the defaults. For example, the below two configuration files are equivalent.

```yaml
# default batch_size is 32, so this line is redundant
batch_size: 32
learning_rate: 0.000001
```

or

```yaml
learning_rate: 0.000001
```

`main.default.yaml` includes a commented description of what each parameter does.

## Example

1. Train model using (optional) `main.yaml` as configuration file.

   `$ python main.py`

2. Generate hamiltonian tensor and output to `generated.pt` using `model_path` specified in configuration.

   `$ python main.py --generate generated.pt`

3. Train model and output batch statistics

   `$ python main.py -v`

4. Train model and output final statistics as json

   `$ python main.py --json`