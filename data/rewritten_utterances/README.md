# Rewritten utterances

This directory collects the files with utterances rewritten automatically by the approaches presented in [Mele2021].

For the rewriting, we used the labels predicted with the BERT classification at Step 1 and BERT context classification at Step 2 (files `*_S1_BERT_S2_BERT_Context.tsv`) as well as the ground-truth labels (files `*_GT.tsv`).

## Methods and corresponding files

## _Standard method_

If u_i is classified as FT, it is enriched with context extracted from the first utterance. If u_i is classified as PT, it is enriched with context from the previous utterance.

- `classif_results_strategy_Standard_S1_BERT_S2_BERT_Context.tsv`: using the predicted labels.

- `classif_results_strategy_Standard_GT.tsv`: using the ground-truth labels.

## _Enriched method_ 

Similar to Standard, but if u_i is classified as PT, it is enriched with context from the previous enriched utterance.

- `classif_results_strategy_Enriched_S1_BERT_S2_BERT_Context.tsv`: using the predicted labels.

- `classif_results_strategy_Enriched_GT.tsv`: using the ground-truth labels.

## _Last SE method_

Always propagating the context extracted from the last seen SE utterance.

- `classif_results_strategy_LastSE_S1_BERT_S2_BERT_Context.tsv`: using the predicted labels.

- `classif_results_strategy_LastSE_GT.tsv`: using the ground-truth labels.

## _First and last SE method_

Similar to Last SE,  but the context is extracted  from both the first utterance and the last seen SE utterance.

- `classif_results_strategy_FirstAndLastSE_S1_BERT_S2_BERT_Context.tsv`: using the predicted labels.

- `classif_results_strategy_FirstAndLastSE_GT.tsv`: using the ground-truth labels.

## _First or last SE method_

Similar to Standard. If u_i is classified as FT, it is enriched with the context extracted from the first utterance. If u_i is classified as PT, then the context is extracted from the last seen SE instead of the previous utterance.

- `classif_results_strategy_FirstOrLastSE_S1_BERT_S2_BERT_Context.tsv`: using the predicted labels.

- `classif_results_strategy_FirstOrLastSE_GT.tsv`: using the ground-truth labels.


## Format 

utteranceId \t utterance 

## References

[Mele2021] Ida Mele, Cristina Ioana Muntean, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Ophir Frieder, Adaptive utterance rewriting for conversational search, IProcessing & Management, Volume 58, Issue 6, 2021, https://doi.org/10.1016/j.ipm.2021.102682.

