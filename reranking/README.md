# Reranking

For reranking we use the model proposed by Nogueira and Cho [Nogueira2019] trained on MS-MARCO.

In the notebook `Reranking with dl4marco-bert.ipynb` we show the steps taken for the reranking phase:
1. converting the retrieval output into the format required by the model
2. converting the input files into TF records using the `convert_CAST_to_tfrecord.sh` script
3. run the model in inference over the test set using the `test_CAST.sh` script
4. use trec_eval for evalution


## References
[Nogueira2019] Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. ArXiv, abs/1901.04085.
