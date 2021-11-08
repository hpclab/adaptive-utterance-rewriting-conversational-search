import spacy
from smartpipeline.stage import Stage, DataItem

nlp = spacy.load("en_core_web_sm")


class Topic(Stage):
    def __init__(self):
        self.first_topic = ""
        self.current_topic = ""
        self.context_list = []
        self.pos_list = ["nsubj", "dobj", "pobj"]
        self.third_person_prons = ["he", "she", "it", "they", "him", "her",
                                   "them", "his", "her", "its", "their"]
        self.cue_phrases = ["tell me more about", "tell me about"]

    def _find_first_topic(self, doc):
        # GET the first topic
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in self.pos_list:
                if chunk.root.pos_ != "PRON":
                    self.first_topic += " " + chunk.text

        # NO FIRST TOPIC
        if self.first_topic == "":
            for token in doc:
                if token.pos_ not in ["VERB", "PUNCT"]:
                    self.first_topic += token.text + " "

        self.current_topic = self.first_topic

    def _find_current_topic(self, utt, doc):
        # CUE
        pron = False
        for cue in self.cue_phrases:
            if cue in utt.lower():
                # check if pron:
                for token in doc:
                    if (token.tag_ == "PRP" or token.tag_ == "PRP$") and \
                            token.text in self.third_person_prons:
                        pron = True
                if not pron:
                    self.current_topic = utt.lower().replace(cue, "")\
                                                    .replace(".", "")

    def _add_to_context(self, doc):
        # add to context list
        for chunk in doc.noun_chunks:
            # could add dobj pobj
            if ("nsubj" in chunk.root.dep_
                or "pobj" in chunk.root.dep_
                or "dobj" in chunk.root.dep_) \
                    and chunk.root.pos_ != "PRON":
                self.context_list.append(chunk.text)

    def _add_current_utt_context(self, doc):
        # add to context list
        doc_context = ""
        for chunk in doc.noun_chunks:
            # could add dobj pobj
            if ("nsubj" in chunk.root.dep_
                or "pobj" in chunk.root.dep_
                or "dobj" in chunk.root.dep_) \
                    and chunk.root.pos_ != "PRON":
                doc_context += chunk.text + " "

        if doc_context != "":
            self.context_list.append(doc_context)

        if len(self.context_list) > 2:
            newlist = self.context_list[-2:]
            self.context_list = newlist

    def _rewrite_utt(self, doc, trailing=False, context=False):
        new_utt = ""
        for token in doc:
            if (token.tag_ == "PRP" or token.tag_ == "PRP$") \
                    and token.text in self.third_person_prons:
                new_utt += self.current_topic + " "
            else:
                new_utt += token.text + " "

        # TRAILING THE TOPIC
        if trailing:
            if self.current_topic.lower() not in new_utt.lower():
                new_utt += self.current_topic + " "
            if self.first_topic.lower() not in new_utt.lower():
                new_utt += self.first_topic

        # TRAILING THE CONTEXT
        if context and len(self.context_list) > 0:
            new_utt += " ".join(self.context_list)

        return new_utt


class FirstTopic(Topic):
    def process(self, item: DataItem) -> DataItem:
        qid = item.payload["id"]
        utt: str = item.payload["utterance"]
        doc = nlp(utt)

        if qid.endswith("_1"):
            self.first_topic = ""
            self._find_first_topic(doc)
        else:
            utt = self._rewrite_utt(doc, trailing=True)

        item.payload["utterance"] = utt
        return item


class TopicShift(Topic):
    def process(self, item: DataItem) -> DataItem:
        qid = item.payload["id"]
        utt: str = item.payload["utterance"]
        doc = nlp(utt)

        if qid.endswith("_1"):
            self.first_topic = ""
            self.current_topic = ""
            self._find_first_topic(doc)
        else:
            self._find_current_topic(utt, doc)
            utt = self._rewrite_utt(doc, trailing=True)

        item.payload["utterance"] = utt
        return item


class KeepContext(Topic):
    def process(self, item: DataItem) -> DataItem:
        qid = item.payload["id"]
        utt: str = item.payload["utterance"]
        doc = nlp(utt)

        if qid.endswith("_1"):
            self.first_topic = ""
            self.current_topic = ""
            self.context_list = []
            self._find_first_topic(doc)
            self._add_to_context(doc)
        else:
            self._find_current_topic(utt, doc)
            self._add_to_context(doc)
            utt = self._rewrite_utt(doc, trailing=False, context=True)

        item.payload["utterance"] = utt
        return item


class PrevContext(Topic):
    def process(self, item: DataItem) -> DataItem:
        qid = item.payload["id"]
        utt: str = item.payload["utterance"]
        doc = nlp(utt)

        if qid.endswith("_1"):
            self.first_topic = ""
            self.current_topic = ""
            self.context_list = []
            self._find_first_topic(doc)
            self._add_current_utt_context(doc)
        else:
            self._find_current_topic(utt, doc)
            self._add_current_utt_context(doc)
            utt = self._rewrite_utt(doc, trailing=True, context=True)

        item.payload["utterance"] = utt
        return item