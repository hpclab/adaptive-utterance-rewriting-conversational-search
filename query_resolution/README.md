# Query resolution

In this module we implement the ways in which we do query rewriting describing the methods proposed in [Mele2020] and [Mele2021].

## 1. Supervised

This submodule implements how query rewriting is done in [Mele2021], based on the classification labels for each utterance. 

In `topic_utils.py` we implement multiple methods to extract topics and how the actual rewriting is done. These methods are later used in the notebook below.

In the `Classification-S1-BERT-S2-BERTContext.ipynb` notebook we give an example of how we mix together the two utterance classification steps and the perform rewriting (with 5 different strategies: **Standard, Enriched, Last SE, First and last SE, First or last SE**) given the predicted classes. To speed up the process we use the predictions from using the two models (if run on a GPU the actual models could be used in inference, instead of the predicted labels), rather than the models themselves, in an offline way. The notebook allows also visualization of results and computes preliminary evaluation metrics. 

The notebook corresponds to the best model combination, namely for Step 1 (self-explanatory vs not self-explanatory), we use a plain BERT model and for Step 2 (first topic vs previous topic) we use the BERT context model. See [Mele2021] for more details.



## 2. Unsupervised

This submodule implements the methods described in [Mele2020], which are now used as baselines in the journal version of the paper [Mele2021].

The files `source.py`, `topic.py` and `coreference.py` implement the helper classes we use in each pipeline to generate the rewriting:
- `pipeline_coref1.py` - uses neural coreferencing model 
- `pipeline_cored2.py` - uses AllenNLP coreferencing model
- `pipeline_first_topic.py` - implements our FirstTopic method 
- `pipeline_topic_shift.py` - implements our TopicShift method

When we want to generate the rewriting of queries we simply run the desired pipeline. The file will write in output the rewritten queries from the input, the **CAST 2019** utterance test set. If you want to change the input file, simply edit the new source inside the script.

```
python pipeline_topics_shift.py
```

## References

- [Mele2020] Ida Mele, Cristina Ioana Muntean, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, and Ophir Frieder. 2020. Topic Propagation in Conversational Search. SIGIR '20.2057–2060. DOI:https://doi.org/10.1145/3397271.3401268

- [Mele2021] Ida Mele, Cristina Ioana Muntean, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Ophir Frieder, Adaptive utterance rewriting for conversational search, IProcessing & Management, Volume 58, Issue 6, 2021, https://doi.org/10.1016/j.ipm.2021.102682.
