# iBRNet

This repository contains the code for performing model training using Elemental Fraction (EF) as the model input.

## Installation Requirements

The basic requirements for using the files are listed in `requirements.txt`.

## Source Files

Use iBRNet.py file to train the iBRNet model.

## Running the code

You can simply run the code (after defining the data path) as follows:

`python iBRNet.py -rlrop 45 -es 50 -sm "model_ibrnet" -prop "target"`

`rlrop` is for Reduce Learning Rate on Platue call back function, 
`es` is for the early stopping call back function,
`sm` is the name of the saved model, 
`prop` is the column to train the model on.
