# Stylometry Classification

###### There are two approaches implemented here that can be followed to classify after feature extraction in the stylometry problem First, Classical Approaches
- Naive Bayes
- Logistic Regression
###### Second, Multi-Layer Perceptron whose architecture is 3 blocks
- Block 1: Linear_Layer(input_dim,256), BatchNorm, ReLU
- Block 2: Linear_Layer(256,128), BatchNorm, ReLU
- Block 3: Linear_Layer(128,2), Sigmoid
## Dataset
data.csv​ : csv file contains text written by two writers
- HP Lovecraft (HPL)
- Mary Wollstonecraft Shelley (MWS)
obtained from [Spooky Author Identification Competition](https://www.kaggle.com/c/spooky-author-identification/) on Kaggle


## Usage

-   Fit Classical Approaches:
    ```bash
    python classical_main.py -classifier [classifier_type] -feature [feature_type]
    ```
    ```console
    classifier_type​ : choose one classifier from (logistic_regression, naive_bayes),
    default is naive_bayes

    feature_type​ : features that will be extracted before classification (TfIdf, NGrams,
    TfIdf_NGrams), default is NGrams
    ```

-   Train Multi-Layer Perceptron:
    ```bash
    python MLP_main.py -feature [feature_type] -BCE [BCE_select] -MSE [MSE_select] -Contrastive [euclidean_contrastive_loss_select] -ContrastiveMargin [contrastive_loss_margin]
    ```
    ```console
    feature_type​ : features that will be extracted before classification (TfIdf, NGrams, TfIdf_NGrams), default is NGrams

    BCE_select​ : 1 if you want to include BCE Loss, 0 otherwise, default is 1

    MSE_select​ : 1 if you want to include MSE Loss, 0 otherwise, default is 0

    euclidean_contrastive_loss_select​ : 1 if you want to include Euclidean Contrastive Loss, 0 otherwise, default is 0

    contrastive_loss_margin​ : margin used in contrastive loss, default is 1
    ```
