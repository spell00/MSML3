
## Authors

* Simon Pelletier (2021-) (Current Maintainer)

# Prerequisites on ubuntu
`apt-get install -y parallel`<br/>
`apt-get install -y python3`<br/>
`apt-get install -y python3-pip`<br/>
`apt-get install -y r-base`<br/>
`apt-get purge -y openjdk-\*`<br/>
`apt install -y openjdk-8-jre`<br/>
`apt install -y openjdk-8-jdk`<br/>
`apt-get install -y ant`<br/>
`apt-get install -y ca-certificates-java`<br/>
`update-ca-certificates -f`<br/>
`chmod +x mzdb2train.sh`<br/>

`chmod +x msml/scripts/mzdb2tsv/amm`

# Install python dependencies
`pip install -r requirements.txt`


# Time summary
Using 200spd lc-msms,
raw2mzdb: ~ ? minutes / sample (Fully parallelized)
mzdb2tsv: ~ ? minutes / sample (Fully parallelized) (depends on the binning)
tsv2df: ~ ? minutes / sample (Fully parallelized) TODO use peak picking?

raw2mzml (using msconvert): ~ 18 minutes / bacterium, 8 minutes / blk (Fully parallelized) (12 files in parallel; 8 minutes when converting a single file)  # TODO msconvert gui doesnt calculate the time it takes. Use the command line version to calculate the average time per bacterium
mzml2csv: ~ 2h30 / sample (Fully parallelized)

On Windows:
The first step needs to be executed on Windows because it calls raw2mzdb.exe and the software only exists for Windows.

In a  Windows PowerShell:


`./msml/preprocess/raw2mzdb.bat`

`./msml/preprocess/raw2mzdb.bat old_data 1 200 0.2 20 0.2 20 0 mutual_info_classif "eco,sag,efa,kpn,blk,pool" 1 0`

The resulting mzdb files are stored in `../../resources/mzdb/$spd/$group/`

On Linux (tested with WLS Ubuntu 20.04):

`bash ./msml/preprocess/mzdb2tsv.sh $mz_bin $rt_bin $spd $group`

The resulting tsv files are stored in `../../resources/mzdb/$spd/$group/`

## Train deep learning model
Command line example:

`python3 msml\dl\train\mlp\train_ae_classifier.py --triplet_loss=1 --predict_tests=1 --dann_sets=0 --balanced_rec_loader=0 --dann_batches=0 --zinb=0 --variational=0 --use_valid=1 --use_test=1`

For your data to work, it should be a matrix: rows are samples, columns are features. Feature names can be whatever,
but the row names (in the first column named ID), the names should be as such: `{experiment_name}_{class}_{batch_number}_{id}`

*** The batch number should start with the letter `p`, followed by batch number. This is because for the experiment
it was designed for, the batches were the plates in which the bacteria grew. It should change soon!
e.g.: `rd159_blk_p16_09`

## Observe results from a server on a local machine 
On local machine:<br/>
`ssh -L 16006:127.0.0.1:6006 simonp@192.168.3.33`

On server:<br/>
`python3 -m tensorboard.main --logdir=/path/to/log/file`

Open in browser:<br/>
`http://127.0.0.1:16006/`

![](E:\GITLAB\MSML\images\ae-dann.png "Autoencoder-DANN")

## Hyperparameters
    thres (float): Threshold for the minimum number of 0 tolerated for a single feature. 
                   0.0 <= thres < 1.0
    dropout (float): Number of neurons that are randomly dropped out. 
                     0.0 <= thres < 1.0
    smoothing (float): Label smoothing replaces one-hot encoded label vector 
                       y_hot with a mixture of y_hot and the uniform distribution:
                       y_ls = (1 - α) * y_hot + α / K
    margin (float): Margin for the triplet loss 
                    (https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905)
    gamma (float): Controls the importance given to the batches adversarial loss
    beta (float): Controls the importance given to the Kullback-Leibler loss
    zeta (float): Controls the importance given to the ZINB loss
    nu (float): Controls the importance given to the ZINB loss
    layer1 (int): The number of neurons the the first hidden layer of the encoder and the
                  last hidden layer of the decoder
    layer2 (int): The number of neurons the the second hidden layer (Bottleneck)
    ncols (int): Number of features to keep
    lr (float): Model's optimization learning rate
    wd (float): Weight decay value
    scale (categorical): Choose between ['none', 'binarize', 'robust', 'standard', 'l1']
    dann_sets (boolean): Use a DANN on set appartenance?
    dann_batches (boolean): USe a DANN on 
    zinb (boolean): Use a zinb autoencoder?
    variational (boolean): Use a variational autoencoder?
    tied_w (boolean): Use Autoencoders with tied weights?
    pseudo (boolean): Use pseudo-labels?
    tripletloss (boolean): Use triplet loss?
    train_after_warmup (boolean): Train the autoencoder after warmup?

## Metrics:
Rec Loss (Reconstruction loss)
Domain Loss: Should be random (e.g. ~0.6931 if 2 batches)
Domain Accuracy: Should be random (e.g. ~0.5 if 2 batches)
(Train, Valid or Test) Loss (l, h or v): Classification loss for low (l), high (h) or very high (v) concentrations
                                         of bacteria in the urine samples. These are the subcategories for the 
                                         data in example_resources, but it might be different if other subcategories 
                                         are different (subcategories are optional).
(Train, Valid or Test) Accuracy (l, h or v): Classification accuracies
(Train, Valid or Test) MCC (l, h or v): Matthews correlation coefficients

## REMARKS
The controls are hard-coded to be named "blanc". That should be change to allow custom names, but that's how it is for the moment