# Classification

The classification module preps all the features and trains the models for the two-step classification.
We use both featured-based models and BERT-based models.

The `utterance_features.py` and `conversation_features.py` implement the features used for the GBDT models, while `generate_features.py` implements the methods that enable the generation of those features. The methods ate then used in the corresponding notebooks.

Our utterance classification methodology involves two cascading steps:
- Step 1 classifies whether and utterance is self-explanatory or not (SE vs. non-SE)
- Step 2 classifies non self-explanatory into first topic or previous topic related utterances (FT vs. PT) 

Notebooks for GBDT based models (using [lightgbm](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)):
- `1.Features Step1.ipynb` loads the dataset and then generates the features for the GBDT model. The feature files are saved in `/data/gbdt_features/` for both training and test sets.
- `2.Features Step2.ipynb` generates features for Step 2 of the classification. Features at Step 2 depend on the labels/predictions at Step 1. According to the model model used for Step 1 we need to generate the features for Step 2 based on the predicted labels. The features generated in isolation are based on the ground truth labels.
- `3.Classification Step1.ipynb` and `4.Classification Step2.ipynb`  loads the datasets, the corresponding features, trains the models and does a preliminary evaluation. The models are saved in `/data/gbdt_models/`.