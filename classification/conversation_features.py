import torch
from transformers import BertTokenizer, BertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc')
model = BertForSequenceClassification.from_pretrained('bert-base-cased-finetuned-mrpc')

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
model2 = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
#https://medium.com/genei-technology/richer-sentence-embeddings-using-sentence-bert-part-i-ce1d9e0b1343

pos_list = ["nsubj", "dobj", "pobj"]


def _find_topic(doc):
    topic = ""
    
    # GET the first topic
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ in pos_list:
            if chunk.root.pos_ != "PRON":
                topic += " " + chunk.text

    # NO FIRST TOPIC
    if topic == "":
        return str(doc)

    return topic


def noun_chunks_cosine_similarity_first(df, feature_df):
    """
    This is done conversation wise. We need utt_id and utterance.
    df is usually the train_df or test_df.
    Computing similarity between current utterance and the first 
    of that conversation.
    We return a dictionary of utt_id and feature values.
    We join results to the feature_df by mapping the dict values 
    with the keys: utt_id
    
    test_feat["cosine_first"] = test_feat[0].map(cosine_first)
    """
    utt_dict = dict(zip(df[0], df[1]))
    feat_dict = {}
    for utt_id, current_utt in utt_dict.items():
        first_utt_id = utt_id.split("_")[0] + "_1"
        
        row_index_first = df.index[df[0] == first_utt_id].tolist()[0]
        row_index_current = df.index[df[0] == utt_id].tolist()[0]
        
        first_topic = _find_topic(feature_df.at[row_index_first, "doc"])
        current_topic = _find_topic(feature_df.at[row_index_current, "doc"])
    
        current_embed = model2.encode(current_topic)
        first_embed = model2.encode(first_topic)

        similarity = 1 - cosine(current_embed[0], first_embed[0])
        feat_dict[utt_id] = similarity
        
    return feat_dict


def noun_chunks_cosine_similarity_previous(df, feature_df):
    """
    This is done conversation wise. We need utt_id and utterance.
    df is usually the train_df or test_df.
    Computing similarity between current utterance and the first 
    of that conversation.
    We return a dictionary of utt_id and feature values.
    We join results to the feature_df by mapping the dict values 
    with the keys: utt_id
    
    test_feat["cosine_prev"] = test_feat[0].map(cosine_prev)
    """
    
    utt_dict = dict(zip(df[0], df[1]))
    feat_dict = {}
    
    for utt_id, current_utt in utt_dict.items():
        conv_id, turn = utt_id.split("_")

        if int(turn) == 1:
            feat_dict[utt_id] = 0.0
        else:
            prev_utt_id = conv_id + "_" + str(int(turn)-1)
            
            row_index_previous = df.index[df[0] == prev_utt_id].tolist()[0]
            row_index_current = df.index[df[0] == utt_id].tolist()[0]
            
            previous_topic = _find_topic(feature_df.at[row_index_previous, "doc"])
            current_topic = _find_topic(feature_df.at[row_index_current, "doc"])
            
            current_embed = model2.encode(current_topic)
            prev_embed = model2.encode(previous_topic)

            similarity = 1 - cosine(current_embed[0], prev_embed[0])
            feat_dict[utt_id] = similarity
        
    return feat_dict


def utterance_cosine_similarity_first(df):
    """
    This is done conversation wise. We need utt_id and utterance.
    df is usually the train_df or test_df.
    Computing similarity between current utterance and the first 
    of that conversation.
    We return a dictionary of utt_id and feature values.
    We join results to the feature_df by mapping the dict values 
    with the keys: utt_id
    
    test_feat["cosine_first"] = test_feat[0].map(cosine_first)
    """
    utt_dict = dict(zip(df[0], df[1]))
    feat_dict = {}
    for utt_id, current_utt in utt_dict.items():
        first_utt_id = utt_id.split("_")[0] + "_1"
        
        current_embed = model2.encode([current_utt])
        first_embed = model2.encode([utt_dict[first_utt_id]])

        similarity = 1 - cosine(current_embed[0], first_embed[0])
        feat_dict[utt_id] = similarity
        
    return feat_dict


def utterance_cosine_similarity_previous(df):
    """
    This is done conversation wise. We need utt_id and utterance.
    df is usually the train_df or test_df.
    Computing similarity between current utterance and the previous 
    of that conversation.
    
    We return a dictionary of utt_id and feature values.
    We join results to the feature_df by mapping the dict values 
    with the keys: utt_id
    
    test_feat["cosine_prev"] = test_feat[0].map(cosine_prev)

    Special case, if current is first there is no previous so feature makes
    no sense (would be as indicative as turn). We return similarity 0.
    """

    utt_dict = dict(zip(df[0], df[1]))
    feat_dict = {}
    
    for utt_id, current_utt in utt_dict.items():
        conv_id, turn = utt_id.split("_")

        if int(turn) == 1:
            feat_dict[utt_id] = 0.0
        else:
            prev_utt_id = conv_id + "_" + str(int(turn)-1)
            
            current_embed = model2.encode([current_utt])
            prev_embed = model2.encode([utt_dict[prev_utt_id]])

            similarity = 1 - cosine(current_embed[0], prev_embed[0])
            feat_dict[utt_id] = similarity
        
    return feat_dict


def is_next_sentence_to_first_neural(df):
    """
    We use a model trained for Sequence Classification:
    BertForSequenceClassification.from_pretrained('bert-base-uncased')

    Other options for tuned for paraphrasing:  'bert-base-cased-finetuned-mrpc'
    
    We return a dictionary of utt_id and feature values.
    We join results to the feature_df by mapping the dict values 
    with the keys: utt_id
    
    test_feat["is_next_first"] = test_feat[0].map(is_next_first)

    :return:
    """
    model.eval()
    utt_dict = dict(zip(df[0], df[1]))
    feat_dict = {}
    
    for utt_id, current_utt in utt_dict.items():
        first_utt_id = utt_id.split("_")[0] + "_1"
        
        is_next = tokenizer.encode_plus(utt_dict[first_utt_id], current_utt, return_tensors="pt")
        is_next_classification_logits = model(**is_next)[0]

        is_next_results = torch.softmax(is_next_classification_logits, dim=1).tolist()[0]
        feat_dict[utt_id] = is_next_results[1]

    return feat_dict  # second elem of the logit


def is_next_sentence_to_previous_neural(df):
    """
    We use a model trained for Sequence Classification:
    BertForSequenceClassification.from_pretrained('bert-base-uncased')
    BertForNextSentencePrediction

    Other options for tuned for paraphrasing:  'bert-base-cased-finetuned-mrpc'
    
    We return a dictionary of utt_id and feature values.
    We join results to the feature_df by mapping the dict values 
    with the keys: utt_id
    
    test_feat["is_next_prev"] = test_feat[0].map(is_next_prev)

    Special case, if current is first there is no previous so feature makes
    no sense (would be as indicative as turn). We return similarity 0.
    :return:
    """

    model.eval()
    utt_dict = dict(zip(df[0], df[1]))
    feat_dict = {}
    
    for utt_id, current_utt in utt_dict.items():
        conv_id, turn = utt_id.split("_")
        
        if int(turn) == 1:
            feat_dict[utt_id] = 0.0
        else:
            prev_utt_id = conv_id + "_" + str(int(turn)-1)
        
            is_next = tokenizer.encode_plus(utt_dict[prev_utt_id], current_utt, return_tensors="pt")
            is_next_classification_logits = model(**is_next)[0]

            is_next_results = torch.softmax(is_next_classification_logits, dim=1).tolist()[0]
            feat_dict[utt_id] = is_next_results[1]

    return feat_dict  # second elem of the logit


def compute_is_next_of_SE_train(df):
    """
    Function used only for training data, it uses the GT labels (SE or noSE)
    This feature is binary:
    0: the previous utterance is not an SE
    1: the previous utterance is an SE
    """
    
    feat_dict_train = {}
    #inizialization (all 0 at the beginning)
    for i in range(0, len(df)):
        utt_id = df.loc[i, 0]
        prev_SE = 0 
        feat_dict_train[utt_id] = prev_SE

    last_SE_seen = 0

    for i in range(1, len(df)):   
        utt_id = df.loc[i,0]
        turn = utt_id.split("_")[1]
        #check if 1st turn of the conversation 
        #(no prev SE, distance from itself 0, it is the last SE seen)
        if turn == "1":
            last_SE_seen = i
        else:    
            conv_id = df.loc[i, 0].split("_")[0]
            conv_id_prev = df.loc[i-1, 0].split("_")[0]

            if conv_id == conv_id_prev: #belong to the same conversation
                if df.loc[i, 2] == "SE":  # the current one is an SE by itself
                    last_SE_seen = i
                elif df.loc[i-1, 2] == "SE":  # the previous one is an SE
                    prev_SE = 1
                    feat_dict_train[utt_id] = prev_SE

    return feat_dict_train


def compute_dist_last_SE_train(df):
    """
    Function used only for training data, it uses the GT labels (SE or noSE)
    This feature is the distance from the last SE seen
    Always 0 if the utterance is an SE itself
    """
    
    feat_dict_distance_train = {}
    #inizialization (all 0 at the beginning)
    for i in range(0, len(df)):
        utt_id = df.loc[i,0]
        dist_last_SE = 0.0
        feat_dict_distance_train [utt_id] = dist_last_SE

    last_SE_seen = 0

    for i in range(1, len(df)):    
        utt_id = df.loc[i, 0]
        turn = utt_id.split("_")[1]
        #check if 1st turn of the conversation 
        #(no prev SE, distance from itself 0, it is the last SE seen)
        if turn == "1":
            last_SE_seen = i
            dist_last_SE = 0.0
            feat_dict_distance_train[utt_id] = dist_last_SE

        else:    
            conv_id = df.loc[i, 0].split("_")[0]
            conv_id_prev = df.loc[i-1, 0].split("_")[0]

            if conv_id == conv_id_prev: #belong to the same conversation
                if df.loc[i, 2] == "SE": # the current one is an SE by itself 
                    last_SE_seen = i
                dist_last_SE = float(i - last_SE_seen)
                feat_dict_distance_train[utt_id] = dist_last_SE

    return feat_dict_distance_train


def compute_is_next_of_SE_test(df, y_pred):
    
    """
    Function used only for test data, it uses the predictions (SE or noSE) of
    classification step 1
    This feature is binary:
    0: the previous utterance is not an SE
    1: the previous utterance is an SE
    """
        
    feat_dict_test = {}
    #this is to make sure that all values are initialized 
    for i in range(0, len(df)):
        utt_id = df.loc[i,0]
        prev_SE = 0 
        feat_dict_test[utt_id] = prev_SE

    # Read the results of the first classification in y_pred
    # Since y_pred vector does not have 1st turn utterance anymore we need
    # to check the indexes and make sure
    # they are aligned to the ones in df (test set)
    for i in range(0, len(df)):     
        if i == 0: 
            #if here, it is the 1st utterance of the 1st conversation
            last_SE_seen = 0
            index_y_pred = 0 #set the index for y_pred 
        else:
            utt_id = df.loc[i, 0]
            turn = utt_id.split("_")[1]
            conv_id = utt_id.split("_")[0]
            conv_id_prev = df.loc[i-1, 0].split("_")[0]

            #check if 1st turn of the conversation
            #(no prev SE, distance from itself = 0, it is the last SE seen)
            if turn == "1":
                last_SE_seen = i
                #df.loc[i, "y_pred"] = y_pred[index_y_pred]
            else:
                #if here it is not 1st turn of the conversation
                if conv_id == conv_id_prev: #just to check that they belong to the same conversation
                    # the current utterance is an SE by itself 
                    if y_pred[index_y_pred] ==  1: 
                        last_SE_seen = i

                    #the current utterance is not a SE, we need to check the previous one
                    else:
                        if index_y_pred == 0:  # only for 2nd utterance in 1st conv, so it is the 1st element in y_pred
                            #if here there is no previous in y_pred, so we need to set the values
                            last_SE_seen = 0
                            feat_dict_test[utt_id] = 1
                        #if here there is a previous in y_pred, so we can check the previous one in y_pred
                        else: 
                            if y_pred[index_y_pred-1] == 1:
                                #previous one is an SE
                                feat_dict_test[utt_id] = 1
                            else:
                                #previous one is not an SE
                                feat_dict_test[utt_id] = 0

                    #y_pred index can move on since it is in the same conversation
                    index_y_pred = index_y_pred + 1
    return feat_dict_test


def compute_dist_last_SE_test(df, y_pred):
    """
    Function used only for test data, it uses the predictions (SE or noSE) of classification step 1 
    This feature is the distance from the last SE seen
    Always 0 if the utterance is an SE itself
    """
    feat_dict_distance_test = {}
    #this is to make sure that all values are initialized 
    for i in range(0, len(df)):    
        utt_id = df.loc[i,0]
        #dist_SE: the number of utterances between the current one and the last SE seen
        dist_last_SE = 0.0
        feat_dict_distance_test [utt_id] = dist_last_SE        
    
    # Read the results of the first classification in y_pred
    # Since y_pred vector does not have 1st turn utterance anymore we need to
    # check the indexes and make sure
    # they are aligned to the ones in df (test set)
    for i in range(0, len(df)):     
        if i == 0: 
            #if here, it is the 1st utterance of the 1st conversation
            last_SE_seen = 0
            index_y_pred = 0 #set the index for y_pred 
        else:
            utt_id = df.loc[i, 0] 
            turn = utt_id.split("_")[1]
            conv_id = utt_id.split("_")[0]
            conv_id_prev = df.loc[i-1, 0].split("_")[0]

            #check if 1st turn of the conversation
            #(no prev SE, distance from itself = 0, it is the last SE seen)
            if turn == "1":
                last_SE_seen = i
            else:
                #if here it is not 1st turn of the conversation
                if conv_id == conv_id_prev: #just to check that they belong to the same conversation
                    # the current utterance is an SE by itself 
                    if y_pred[index_y_pred] == 1:
                        last_SE_seen = i

                    #the current utterance is not a SE, we need to check the previous one
                    else:
                        if index_y_pred == 0:  # only for 2nd utterance in 1st conv, so it is the 1st element in y_pred
                            #if here there is no previous in y_pred, so we need to set the values
                            last_SE_seen = 0
                            dist_last_SE = 1.0
                            feat_dict_distance_test[utt_id] = dist_last_SE

                        #if here there is a previous in y_pred, so we can check the previous one in y_pred
                        else: 
                            if y_pred[index_y_pred-1] == 1: #previous one is an SE
                                dist_last_SE = 1.0
                                feat_dict_distance_test[utt_id] = dist_last_SE

                            else: #previous one is not an SE
                                dist_last_SE = float(i - last_SE_seen)
                                feat_dict_distance_test[utt_id] = dist_last_SE
                               
                    #y_pred index can move on since it is in the same conversation
                    index_y_pred = index_y_pred + 1
    return feat_dict_distance_test
