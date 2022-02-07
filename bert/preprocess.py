import os
import numpy as np
from sklearn.model_selection import train_test_split


all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))

labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}


def preprocess(num):


    X = []
    y = []

    with open("../train_part/train.partaa", encoding="utf-8") as f:

        # count = 0

        for line in f:

            # one line one data
            line = line.strip()
            # features is a list
            features = line.split("\x01")

            for feature, idx in all_features_to_idx.items():
                # print("feature {} has value {}".format(feature, features[idx]))
                if idx == 0:
                    split_index = features[idx].split()
                    # using map() to perform conversion from str to int
                    split_index = list(map(int, split_index))
                    X.append(split_index)

            new_y = []
            for label, idx in labels_to_idx.items():
                # print("label {} has value {}".format(label, features[idx]))
                new_y.append(features[idx])
            y.append(new_y)

            # count += 1
            # if count == 10000: break

        X = np.asarray(X)
        y = np.asarray(y)

        y1 = y[:,num].tolist()
        # y2 = y[:,1].tolist()
        # y3 = y[:,2].tolist()
        # y4 = y[:,3].tolist()

        for i in range(len(y1)):
            if y1[i] == '':
                y1[i] = 0
            else:
                y1[i] = 1
        
        # for i in range(len(y2)):
        #     if y2[i] == '':
        #         y2[i] = 0
        #     else:
        #         y2[i] = 1

        # for i in range(len(y3)):
        #     if y3[i] == '':
        #         y3[i] = 0
        #     else:
        #         y3[i] = 1

        # for i in range(len(y4)):
        #     if y4[i] == '':
        #         y4[i] = 0
        #     else:
        #         y4[i] = 1

        X_train, X_val, y_train, y_val = train_test_split(X, y1, test_size=0.3, random_state=42)
        # X_train2, X_val2, y_train2, y_val2 = train_test_split(X, y2, test_size=0.3, random_state=42)
        # X_train3, X_val3, y_train3, y_val3 = train_test_split(X, y3, test_size=0.3, random_state=42)
        # X_train4, X_val4, y_train4, y_val4 = train_test_split(X, y4, test_size=0.3, random_state=42)
        # print(X_train, y_train)
    return X_train, X_val, y_train, y_val

def get_test():

    X = []

    with open("../val.tsv", encoding="utf-8") as f:

        # count = 0

        for line in f:

            # one line one data
            line = line.strip()
            # features is a list
            features = line.split("\x01")

            for feature, idx in all_features_to_idx.items():
                # print("feature {} has value {}".format(feature, features[idx]))
                if idx == 0:
                    split_index = features[idx].split()
                    # using map() to perform conversion from str to int
                    split_index = list(map(int, split_index))
                    X.append(split_index)

            # count += 1
            # if count == 10000: break

    return X


   