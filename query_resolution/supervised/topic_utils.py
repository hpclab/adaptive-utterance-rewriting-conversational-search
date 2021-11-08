import spacy
nlp = spacy.load("en_core_web_sm")
third_person_prons = ["he", "she", "it", "they", "him", "her", "them", "his", "her", "its", "their"]


def create_doc(utt):
    return nlp(utt)


def _rewrite_utt(doc, first_topic="", previous_topic="",
                 context_list=None, trailing=False):
        new_utt = ""
        for token in doc:
            if (token.tag_ == "PRP" or token.tag_ == "PRP$") \
                    and token.text in third_person_prons:
                if previous_topic != "":
                    new_utt += previous_topic + " "
                if first_topic != "":
                    new_utt += first_topic + " "
            else:
                new_utt += token.text + " "

        # TRAILING THE TOPIC
        if trailing:
            if previous_topic.lower() not in new_utt.lower():
                new_utt += previous_topic + " "
            if first_topic.lower() not in new_utt.lower():
                new_utt += first_topic

        # TRAILING THE CONTEXT
        if context_list is not None:
            new_utt += " ".join(context_list)

        return new_utt


def _find_cue_topic(doc):
    cue_phrases = ["tell me more about", "tell me about"]
    third_person_prons = ["he", "she", "it", "they", "him", "her", "them",
                          "his", "her", "its", "their"]

    current_topic = ""
    pron = False
    for cue in cue_phrases:
        if cue in str(doc).lower():
            # check if pron:
            for token in doc:
                if (token.tag_ == "PRP" or token.tag_ == "PRP$") and \
                        token.text in third_person_prons:
                    pron = True
            if not pron:
                current_topic = str(doc).lower().replace(cue, "").replace(".","")
    return current_topic


def _find_topic(doc):
    """
    We remove if "chunk.root.dep_ in pos_list" for GT and winning method to avoid problems from spacy, when it cannot
    analyse the phrase.
    """
    pos_list = ["nsubj", "dobj", "pobj"]
    topic = ""

    # NEW FEATURE
    # check if it's a cue topic first
    cue_topic = _find_cue_topic(doc)
    if cue_topic != "":
        return cue_topic

    # GET the first topic
    for chunk in doc.noun_chunks:
        # if chunk.root.dep_ in pos_list:
            if chunk.root.pos_ != "PRON":
                topic += " " + chunk.text

    # NO FIRST TOPIC - trick for "Describe Uranus."
    if topic == "":
        for token in doc:
            if token.pos_ not in ["VERB", "PUNCT"]:
                topic += token.text + " "

    return topic


def _find_topic_all(doc):
    pos_list = ["nsubj", "dobj", "pobj"]
    topic = ""

    # NEW FEATURE
    # check if it's a cue topic first
    cue_topic = _find_cue_topic(doc)
    if cue_topic != "":
        return cue_topic

    # GET the first topic
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ != "PRON":
            topic += " " + chunk.text

    # NO FIRST TOPIC - trick for "Describe Uranus."
    if topic == "":
        for token in doc:
            if token.pos_ not in ["VERB", "PUNCT"]:
                topic += token.text + " "

    return topic