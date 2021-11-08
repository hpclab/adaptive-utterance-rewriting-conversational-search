import logging
import spacy
import neuralcoref
from allennlp.predictors.predictor import Predictor
from smartpipeline.stage import Stage, DataItem

nlp = spacy.load("en_core_web_sm")


class AllenCoref(Stage):
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.predictor = Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")

    def _allen_coref(self, conversation):
        v = " ".join(conversation)
        new_conv = self.predictor.coref_resolved(v)
        doc = nlp(new_conv)
        new_utt_list = list(doc.sents)
        if len(new_utt_list) == len(conversation):
            return 1, new_utt_list
        else:
            return 0, new_utt_list

    def process(self, item: DataItem) -> DataItem:
        conversation = item.payload["conversation"]
        status, new_conversation = self._allen_coref(conversation)
        if status != 1:
            self._logger.info("Split manually conversation: ",
                              item.payload["id"])
        if new_conversation is not None:
            item.payload["conversation"] = new_conversation
        return item


class NeuralCoref(Stage):
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        neuralcoref.add_to_pipe(nlp)

    def _neural_coref(self, conversation):
        v = " ".join(conversation)
        doc_with_coref = nlp(v)
        new_conv_coref = doc_with_coref._.coref_resolved
        new_doc_coref = nlp(new_conv_coref)
        new_utt_list = list(new_doc_coref.sents)

        if len(new_utt_list) == len(conversation):
            return 1, new_utt_list
        else:
            return 0, new_utt_list

    def process(self, item: DataItem) -> DataItem:
        conversation = item.payload["conversation"]
        status, new_conversation = self._neural_coref(conversation)
        if status != 1:
            self._logger.info("Split manually conversation: ",
                              item.payload["id"])
        if new_conversation is not None:
            item.payload["conversation"] = new_conversation
        return item
