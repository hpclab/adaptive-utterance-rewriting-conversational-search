import spacy
import tagme
# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = "9cea67c1-bc33-40fc-8693-276cd4dfa693-843339462"

nlp = spacy.load("en_core_web_sm")

pos_list = ["nsubj", "dobj", "pobj", "adj"]
third_person_prons = ["he", "she", "it", "they", "him", "her", "them", "his",
                      "her", "its", "their"]

question_kw = ["what", "who", "when", "where", "why", "which", "how"]
question_phrases = ["how much", "how many", "how long"]
question_phrases_2 = ["how about", "what about"]

cue_phrases = ["tell me more about", "tell me about", "give me"]
cue_kw = ["describe"]

example_kw = ["example", "examples"]

comparison_kw = ["most", "more", "least", "less", "better", "worse",
                 "difference", "different", "compare", "comparison", "nor",
                 "between", "among"]


def ner(doc):
    """
    Spacy NER model
    :param doc: nlp("utterance")
    :return:
    """
    return len(doc.ents)


def ner_binary(doc):
    """
    Has NE, using spacy NER
    :param doc:
    :return:
    """
    count = len(doc.ents)
    if count > 0:
        return 1
    return 0


def ner_tagme(utt, threshold=0.0):
    """
    TagMe NER counts
    :param utt:
    :return:
    """
    num_ann = 0
    annotations = tagme.annotate(utt, lang='en')

    # Print annotations with a score higher than 0.1
    for _ in annotations.get_annotations(threshold):
        num_ann += 1
    return num_ann


def ner_tagme_binary(utt, threshold=0.0):
    """
    TagMe NER check if any named entities
    :param utt:
    :return:
    """
    num_ann = ner_tagme(utt, threshold)
    if num_ann >= 1:
        return 1
    else:
        return 0


def pron(doc):
    """
    Counts pronouns
    :param doc:
    :return:
    """
    count_pron = 0
    for token in doc:
        if token.tag_ == "PRP" or token.tag_ == "PRP$":
            count_pron = count_pron + 1
    return count_pron


def pron_binary(doc):
    """
    Has pronouns?
    :param doc:
    :return:
    """
    for token in doc:
        if token.tag_ == "PRP" or token.tag_ == "PRP$":
            return 1
    return 0


def pron_3rd(doc):
    """
    Counts third person pronouns
    :param doc:
    :return:
    """
    count_pron = 0
    for token in doc:
        if (token.tag_ == "PRP" or token.tag_ == "PRP$") \
                and token.text in third_person_prons:
            count_pron += 1
    return count_pron


def pron_3rd_binary(doc):
    """
    Has third person pronouns?
    :param doc:
    :return:
    """
    for token in doc:
        if (token.tag_ == "PRP" or token.tag_ == "PRP$") \
                and token.text in third_person_prons:
            return 1
    return 0


def noun(doc):
    """
    Counts the nouns
    :param doc:
    :return:
    """
    count_noun = 0
    for token in doc:
        if token.pos_ == "NOUN":
            count_noun += 1
    return count_noun


def noun_binary(doc):
    """
    Has nouns?
    :param doc:
    :return:
    """
    for token in doc:
        if token.pos_ == "NOUN":
            return 1
    return 0


def adj(doc):
    """
    Counts adjectives
    :param doc:
    :return:
    """
    count_adj = 0
    for token in doc:
        if token.pos == "ADJ":
            count_adj += 1
    return count_adj


def adj_binary(doc):
    """
    Has adjectives?
    :param doc:
    :return:
    """
    for token in doc:
        if token.pos_ == "ADJ":
            return 1
    return 0


def adj_comp(doc):
    """
    Count comparative or superlative adjectives
    :param doc:
    :return:
    """
    count_adj = 0
    for token in doc:
        if token.tag_ == "JJR" or token.tag_ == "JJR$" or token.tag_ == "JJS" \
                or token.tag_ == "JJS$":
            count_adj = count_adj + 1
    return count_adj


def adj_comp_binary(doc):
    """
    Has comparative or superlative adjectives
    :param doc:
    :return:
    """
    for token in doc:
        if token.tag_ == "JJR" or token.tag_ == "JJR$" or token.tag_ == "JJS" \
                or token.tag_ == "JJS$":
            return 1
    return 0


def adv(doc):
    """
    Count adverbs
    :param doc:
    :return:
    """
    count_adj = 0
    for token in doc:
        if token.pos_ == "ADV":
            count_adj = count_adj + 1
    return count_adj


def adv_binary(doc):
    """
    Has adverbs?
    :param doc:
    :return:
    """
    for token in doc:
        if token.pos_ == "ADV":
            return 1
    return 0


def adv_comp(doc):
    """
    Count comparative or superlative adverbs
    :param doc:
    :return:
    """
    count_adj = 0
    for token in doc:
        if token.tag_ == "RBR" or token.tag_ == "RBR$" or \
                token.tag_ == "RBS" or token.tag_ == "RBS$":
            count_adj = count_adj + 1
    return count_adj


def adv_comp_binary(doc):
    """
    Has comparative or superlative adverbs?
    :param doc:
    :return:
    """
    for token in doc:
        if token.tag_ == "RBR" or token.tag_ == "RBR$" or\
                token.tag_ == "RBS" or token.tag_ == "RBS$":
            return 1
    return 0


def question(doc, kw_to_check):
    """
    Is it a question type?
    :param doc:
    :param kw_to_check: list of keywords
    :return:
    """
    count_question_kw = 0
    for token in doc:
        if token.text.lower() in kw_to_check:
            count_question_kw = count_question_kw + 1
    return count_question_kw


def question_binary(doc, kw_to_check):
    """
    Contains specific words?
    :param doc:
    :param kw_to_check:
    :return:
    """
    
    if str(doc).lower().startswith(kw_to_check[0]):
        return 1
    return 0


def question_phrase(utt, phrases_to_check):
    """
    Checks how many phrases of more than one word (e.g., "how many") in the utt
    :param utt:
    :param phrases_to_check:
    :return:
    """
    count_question_ph = 0
    for token in phrases_to_check:
        if token in utt.lower():
            count_question_ph = count_question_ph + 1
    return count_question_ph


def question_phrase_binary(utt, phrases_to_check):
    """
    Checks if any phrases of more than one word (e.g., "how many") in the utt
    :param utt:
    :param phrases_to_check:
    :return:
    """
    for token in phrases_to_check:
        if token in utt.lower():
            return 1
    return 0


def cue_keyword(doc, kw_to_check):
    """
    Checks for one single words (e.g., "describe" or "example")
    :param doc:
    :param kw_to_check:
    :return:
    """
    num_cue_kw = 0
    for token in doc:
        if token.text.lower() in kw_to_check:
            num_cue_kw = num_cue_kw + 1
    return num_cue_kw


def cue_keyword_binary(doc, kw_to_check):
    """
    Checks if there are one single words (e.g., "describe" or "example")
    :param doc:
    :param kw_to_check:
    :return:
    """
    for token in doc:
        if token.text.lower() in kw_to_check:
            return 1
    return 0


def cue_phrase(utt, phrases_to_check):
    """
    Checks how many phrases of more than one word (e.g., "tell me about")
    :param utt:
    :param phrases_to_check:
    :return:
    """
    num_cue = 0
    for token in phrases_to_check:
        if token in utt.lower():
            num_cue = num_cue + 1
    return num_cue


def cue_phrase_binary(utt, phrases_to_check):
    """
    Cchecks if any phrases of more than one word (e.g., "tell me about")
    :param utt:
    :param phrases_to_check:
    :return:
    """
    for token in phrases_to_check:
        if token in utt.lower():
            return 1
    return 0


def is_first(id):
    """
    Check if it's the first utterance of a conversation
    :param id:
    :return:
    """
    id_turn = id.split("_")[1].strip()
    if str(id_turn) == "1":
        return 1
    else:
        return 0


def what_is_question(doc):
    """
    Validates if it's a simple "what is XXXXX" questions
    # check 1) what, 2) to be verb 3) nsubj (different from 3rd pronoun)
    :param doc:
    :return:
    """
    what_flag = to_be_flag = n_subj_flag = False
    for token in doc:
        if token.text == "What":
            what_flag = True
        if token.dep_ == "ROOT" and token.lemma_ == "be":
            to_be_flag = True
        if token.dep_ == "nsubj":
            if not token.text in third_person_prons:
                n_subj_flag = True

    if what_flag and to_be_flag and n_subj_flag:
        return 1
    else:
        return 0


def what_is_question_2(doc):
    """
    Validates if it's a simple "what is XXXXX" questions
    # check 1) what, 2) to be verb
    3) nsubj (different from 3rd pronoun) 4) num_noun_chuck == 2
    :param doc:
    :return:
    """
    what_flag = to_be_flag = n_subj_flag = False
    for token in doc:
        if token.text == "What":
            what_flag = True
        if token.dep_ == "ROOT" and token.lemma_ == "be":
            to_be_flag = True
        if token.dep_ == "nsubj":
            if not token.text in third_person_prons:
                n_subj_flag = True

    num_chunks = len(list(doc.noun_chunks))

    if what_flag and to_be_flag and n_subj_flag and num_chunks == 2:
        return 1
    else:
        return 0


def what_is_question_3(doc):
    """
    Validates if it's a simple "what is XXXXX" questions
    # check 1) what, 2) to be verb 3) num_noun_chuck == 2
    :param doc:
    :return:
    """
    what_flag = to_be_flag = False
    for token in doc:
        if token.text == "What":
            what_flag = True
        if token.dep_ == "ROOT" and token.lemma_ == "be":
            to_be_flag = True

    num_chunks = len(list(doc.noun_chunks))

    if what_flag and to_be_flag and num_chunks == 2:
        return 1
    else:
        return 0


def tell_me_question(doc):
    """
    Validates if it's a simple "tell me about XXXXX" question
    :param doc:
    :return:
    """
    tell_me_flag = third_pers = False
    cue = str(doc).lower()

    if "Tell me about".lower() in cue or "Tell me more about".lower() in cue:
        tell_me_flag = True

    for token in doc:
        if token.text in third_person_prons:  # ususally pobj
            third_pers = True

    num_chunks = len(list(doc.noun_chunks))

    if tell_me_flag and not third_pers and num_chunks == 2:
        return 2
    elif tell_me_flag and third_pers and num_chunks == 2:
        return 1
    else:
        return 0


def num_noun_chunks(doc):
    """
    Counts how many NOUN CHUNKS (spacy)
    :param doc:
    :return:
    """
    return len(list(doc.noun_chunks))


def complete_sentence(doc):
    """
    Checks if the sentence is complete (very powerful, but may be misleading)
    :param doc:
    :return:
    """
    has_noun = 2
    has_verb = 1
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "PRON"]:
            has_noun -= 1
        elif token.pos_ == "VERB":
            has_verb -= 1
    if has_noun < 1 and has_verb < 1:
        return 1
    return 0


def question_mark(doc):
    """
    Checks if utterance has a question mark
    :param doc:
    :return:
    """
    for token in doc:
        if token.text == "?":
            return 1
    return 0


def question_mark_third_person(doc):
    """
    Checks if utterance has a question mark and at least one third person pronoun
    :param doc:
    :return:
    """
    has_third = pron_3rd_binary(doc) 
    if has_third == 1:        
        for token in doc:
            if token.text == "?":
                return 1
        return 0
    else:
        return 0
