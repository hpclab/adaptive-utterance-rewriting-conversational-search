import pandas as pd
from classification.utterance_features import *
from classification.conversation_features import utterance_cosine_similarity_first, \
    utterance_cosine_similarity_previous, is_next_sentence_to_first_neural, \
    is_next_sentence_to_previous_neural, compute_dist_last_SE_train, \
    compute_is_next_of_SE_train, compute_is_next_of_SE_test, compute_dist_last_SE_test, \
    noun_chunks_cosine_similarity_first, noun_chunks_cosine_similarity_previous

nlp = spacy.load("en_core_web_sm")


def load_utterance_df(trainFile, testFile):
    """
    Loads utterance and utterance ID from train and test file.
    :return:
    """
    train_df = pd.read_csv(trainFile, delimiter="\t", header=None)
    test_df = pd.read_csv(testFile, delimiter="\t", header=None)
    return train_df, test_df


def load_feature_df(train_df, test_df):
    """
    Creates new feature dataframes for each upon which features are added
    :param test_df:
    :param train_df:
    :return:
    """
    train_feat_df = train_df[[0]].copy()
    test_feat_df = test_df[[0]].copy()

    # added a column to the nlp linguistic annotation features
    train_feat_df["doc"] = train_df[1].apply(nlp)
    test_feat_df["doc"] = test_df[1].apply(nlp)

    return train_feat_df, test_feat_df


def add_features_step1(df, feature_df):
    """
    Add features to the feature_df corresponding to TRAIN or TEST.
    :param df:
    :param feature_df:
    :return:
    """
    # utterance features
    feature_df["utt_len"] = df[1].str.len()
    feature_df["num_tokens"] = feature_df["doc"].str.len()
    feature_df["complete_sent"] = feature_df["doc"].apply(complete_sentence)
    feature_df["question_mark"] = feature_df["doc"].apply(question_mark)

    # NER
    feature_df['ner'] = feature_df["doc"].apply(ner)
    feature_df['ner_b'] = feature_df["doc"].apply(ner_binary)

    # NER with TagMe
    feature_df['ner_tm_0'] = df[1].apply(ner_tagme)
    feature_df['ner_tm_1'] = df[1].apply(ner_tagme, threshold=0.1)
    feature_df['ner_tm_b'] = df[1].apply(ner_tagme_binary)

    # nouns
    feature_df['noun'] = feature_df["doc"].apply(noun)
    feature_df['noun_b'] = feature_df["doc"].apply(noun_binary)

    # adjectives
    feature_df['adj'] = feature_df["doc"].apply(adj)
    feature_df['adj_b'] = feature_df["doc"].apply(adj_binary)

    feature_df['adj_comp'] = feature_df["doc"].apply(adj_comp)
    feature_df['adj_comp_b'] = feature_df["doc"].apply(adj_comp_binary)

    # adverbs
    feature_df['adv'] = feature_df["doc"].apply(adv)
    feature_df['adv_b'] = feature_df["doc"].apply(adv_binary)

    feature_df['adv_comp'] = feature_df["doc"].apply(adv_comp)
    feature_df['adv_comp_b'] = feature_df["doc"].apply(adv_comp_binary)

    # pronouns
    feature_df['pron'] = feature_df["doc"].apply(pron)
    feature_df['pron_b'] = feature_df["doc"].apply(pron_binary)

    feature_df['pron_3rd'] = feature_df["doc"].apply(pron_3rd)
    feature_df['pron_3rd_b'] = feature_df["doc"].apply(pron_3rd_binary)

    # cue phases, such as "tell me about" "tell me more about" "give me"
    feature_df['cue_ph'] = df[1].apply(cue_phrase, phrases_to_check=cue_phrases)
    feature_df['cue_ph_b'] = df[1].apply(cue_phrase_binary,
                                         phrases_to_check=cue_phrases)

    # cue keywords, such as "describe" and example or comparison keywords
    feature_df['cue_kw'] = feature_df["doc"].apply(cue_keyword,
                                                   kw_to_check=cue_kw)
    feature_df['cue_kw_b'] = feature_df["doc"].apply(cue_keyword_binary,
                                                     kw_to_check=cue_kw)

    feature_df['cue_ex'] = feature_df["doc"].apply(cue_keyword,
                                                   kw_to_check=example_kw)
    feature_df['cue_ex_b'] = feature_df["doc"].apply(cue_keyword_binary,
                                                     kw_to_check=example_kw)

    feature_df['cue_comp'] = feature_df["doc"].apply(cue_keyword,
                                                     kw_to_check=comparison_kw)
    feature_df['cue_comp_b'] = feature_df["doc"].apply(cue_keyword_binary,
                                                       kw_to_check=comparison_kw)

    # questions (one word, e.g., "what", "when")
    feature_df['question'] = feature_df["doc"].apply(question,
                                                     kw_to_check=question_kw)
    feature_df['question_b'] = feature_df["doc"].apply(question_binary,
                                                       kw_to_check=question_kw)

    # questions (more than one word, e.g., "how many")
    feature_df['question_ph'] = df[1].apply(question_phrase,
                                            phrases_to_check=question_phrases)
    feature_df['question_ph_b'] = df[1].apply(question_phrase_binary,
                                              phrases_to_check=question_phrases)

    # check single question kw
    feature_df['what'] = feature_df["doc"].apply(question_binary,
                                                 kw_to_check=["what"])
    feature_df['where'] = feature_df["doc"].apply(question_binary,
                                                  kw_to_check=["where"])
    feature_df['when'] = feature_df["doc"].apply(question_binary,
                                                 kw_to_check=["when"])
    feature_df['who'] = feature_df["doc"].apply(question_binary,
                                                kw_to_check=["who"])
    feature_df['why'] = feature_df["doc"].apply(question_binary,
                                                kw_to_check=["why"])
    feature_df['which'] = feature_df["doc"].apply(question_binary,
                                                  kw_to_check=["which"])
    feature_df['how'] = feature_df["doc"].apply(question_binary,
                                                kw_to_check=["how"])

    feature_df['how_much'] = df[1].apply(question_phrase_binary,
                                         phrases_to_check=["how much"])
    feature_df['how_many'] = df[1].apply(question_phrase_binary,
                                         phrases_to_check=["how many"])
    feature_df['how_long'] = df[1].apply(question_phrase_binary,
                                         phrases_to_check=["how long"])

    # new features
    feature_df['what_is'] = feature_df["doc"].apply(what_is_question)
    feature_df['what_is_2'] = feature_df["doc"].apply(what_is_question_2)
    feature_df['what_is_3'] = feature_df["doc"].apply(what_is_question_3)
    feature_df['tell_me_question'] = feature_df["doc"].apply(tell_me_question)
    feature_df['n_chunks'] = feature_df["doc"].apply(num_noun_chunks)

    # questions (more than one word, e.g., "how about", "what about")
    feature_df['question_ph_2'] = df[1].apply(question_phrase,
                                              phrases_to_check=question_phrases_2)
    feature_df['question_ph_2_b'] = df[1].apply(question_phrase_binary,
                                                phrases_to_check=question_phrases_2)

    # ? and it
    feature_df['ques_mark_it'] = feature_df["doc"].apply(
        question_mark_third_person)

    return feature_df


def add_features_step2(df, train_df, test_df, y_pred, feature_df):
    """
    Add features to the feature_df corresponding to TRAIN or TEST, based on
    the df

    :param df:
    :param feature_df:
    :return:
    """

    feature_df["turn"] = feature_df[0].str.split("_").str[1].astype(int)

    cosine_first = utterance_cosine_similarity_first(df)
    feature_df["cosine_first"] = feature_df[0].map(cosine_first)

    cosine_prev = utterance_cosine_similarity_previous(df)
    feature_df["cosine_prev"] = feature_df[0].map(cosine_prev)

    is_next_first = is_next_sentence_to_first_neural(df)
    feature_df["is_next_first"] = feature_df[0].map(is_next_first)

    is_next_prev = is_next_sentence_to_previous_neural(df)
    feature_df["is_next_prev"] = feature_df[0].map(is_next_prev)

    nc_cosine_first = noun_chunks_cosine_similarity_first(df, feature_df)
    feature_df["nc_cosine_first"] = feature_df[0].map(nc_cosine_first)

    nc_cosine_prev = noun_chunks_cosine_similarity_previous(df, feature_df)
    feature_df["nc_cosine_prev"] = feature_df[0].map(nc_cosine_prev)

    is_next_of_SE = compute_is_next_of_SE_train(train_df)
    feature_df["is_next_of_SE"] = feature_df[0].map(is_next_of_SE)

    dist_last_SE = compute_dist_last_SE_train(train_df)
    feature_df["dist_last_SE"] = feature_df[0].map(dist_last_SE)

    # Features that depend on step 1 - only for TEST
    # The function is made for the TEST set since we assume from Step1 we only
    # get 174 judgements
    is_next_of_SE = compute_is_next_of_SE_test(test_df, y_pred)
    feature_df["is_next_of_SE"] = feature_df[0].map(is_next_of_SE)

    dist_last_SE = compute_dist_last_SE_test(test_df, y_pred)
    feature_df["dist_last_SE"] = feature_df[0].map(dist_last_SE)

    return feature_df
