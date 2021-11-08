# Datasets

This directory collects the datasets used in [Mele2021] from TREC Conversational Assistant Track (CAsT) 2019 [Dalton2020] and ConvQuestions (ConvQ) [Christmann2019] datasets. 

## TREC CAsT 2019

A dataset with 80 multi-turn conversations and 748 utterances in total. Relevance judgments, graded on a three-point scale (i.e., 2 very relevant, 1 relevant, and 0 irrelevant), are provided for 194 out of the 748 utterances. These utterances are used as test set.

## ConvQ 

A dataset with 350 conversations, among which we selected 214 conversations for a total of 1,010 utterances. This subset of conversations provide relevant examples of our classification classes for the context identification task. Notice that we used this dataset because the CAsT dataset does not provide enough utterances to train effective classifiers. ConvQ utterances are used only for enriching the training set for the classification task, and are not used for testing the performance of the conversational IR system, as the utterance-passage relevance judgements are missing for them.

## N.B.

ConvQ is used for the training of the classifier. CAsT 2019 data is split into two files one for training and the other one for testing.

- **Training set:** CAsT 2019 conversations from training set and from test set without qrel + ConvQ dataset 

- **Test set:** CAsT 2019 conversations with qrel

## Format

utteranceId \t utterance \t label

convID_turnID	for CAsT-19 dataset (convID <= 80) **??**

convID_turnID	for ConvQuestions dataset (convID >= 81) **??**

## Labels 

- **SE:** classification label for utterances that are Self Explanatory (e.g., they do not need any rewriting)

- **FT:** classification label for utterances referring to the First Topic in the conversation 

- **PT:** classification label for utterances referring to a Previous Topic in the conversation (different from the first topic)

## References

[Mele2021] Ida Mele, Cristina Ioana Muntean, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Ophir Frieder, Adaptive utterance rewriting for conversational search, IProcessing & Management, Volume 58, Issue 6, 2021, https://doi.org/10.1016/j.ipm.2021.102682.

[Dalton2020] Jeffrey Dalton, Chenyan Xiong, Vaibhav Kumar, and Jamie Callan. CAsT-19: A Dataset for Conversational Information Seeking. SIGIR 2020: 1985-1988 ACM. 

[Christmann2019] Philipp Christmann, Rishiraj Saha Roy, Abdalghani Abujabal, Jyotsna Singh, and Gerhard Weikum. Look before You Hop: Conversational Question Answering over Knowledge Graphs Using Judicious Context Expansion. CIKM 2019: 729–738 ACM.


	
