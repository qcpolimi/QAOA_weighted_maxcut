# QAOA_weighted_maxcut

This repository is published in support of publication "Benchmarking QAOA for Max-Cut on selected instances

## Installation

Install the necessary libraries listed in requirements.txt via pip or conda

## Use

Modify the runner by setting the `folder` variable to a local path that will be used to store the results of the QAOA runs (as  csv)

Modify the rest of the runner in order to decide the graph properties and the range of the QAOA parameters

If you do not have a GPU on your workstation, remember to set the attribute `GPU = False` in the `run_qaoa` method.