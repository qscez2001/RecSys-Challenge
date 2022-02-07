import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime


all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
print(all_features_to_idx)

labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}


def train(X, z, saved_model_name):

    z = [0 if _z == '' else 1 for _z in z]
    # print(z)
    # for i in range(len(z)):

        # if z[i] == '':
        #     z[i] = 0
        # else:
        #     z[i] = 1


    print("split data")
    X_train, X_val, y_train, y_val = train_test_split(X, z, test_size=0.2, random_state=42)
   
    print("start training")
    param_dist = {'objective': 'binary:logistic', 'n_estimators':30, 'n_jobs':7}

    clf = xgb.XGBClassifier(**param_dist)

    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='logloss',
            verbose=True)

    evals_result = clf.evals_result()
    print(evals_result)

    # clf = xgb.XGBModel(**param_dist)

    # clf.fit(X_train, y_train,
    #         eval_set=[(X_train, y_train), (X_val, y_val)],
    #         eval_metric='logloss',
    #         verbose=True)

    # evals_result = clf.evals_result()
    # print(evals_result)

    with open(saved_model_name, 'wb') as fd:
        pickle.dump(clf, fd)



with open("train_part/train.partaa", encoding="utf-8") as f:

    # count = 0

    # X, y = [], []
    X = []
    y = []

    for line in f:

        # one line one data
        line = line.strip()
        # features is a list
        features = line.split("\x01")

        new_X = []
        for feature, idx in all_features_to_idx.items():
            # print("feature {} has value {}".format(feature, features[idx]))

            # 'present_links': 4 present_domains: 5
            if idx == 4 or idx == 5:

                if features[idx] != '':
                    features[idx] = 1
                else:
                    features[idx] = 0
                    
                new_X.append(features[idx])
            # 'present_media': 3
            if idx == 3:

                # dic = {'Photo': 0, 'Video': 1, 'Gif': 2, '': 3}

                # features[idx] = dic[features[idx]]

                if features[idx] == 'Photo':
                    features[idx] = 0
                elif features[idx] == 'Video':
                    features[idx] = 1
                elif features[idx] == 'Gif':
                    features[idx] = 2
                else:
                    features[idx] = 3

                # if features[idx] == '':
                #     media_count = [0, 0, 0]
                # else:
                #     token = features[idx].split('\t')
                #     p_num = token.count('Photo')
                #     v_num = token.count('Video')
                #     g_num = token.count('Gif')
                # media_count = [p_num, v_num, g_num]
                

                new_X.append(features[idx])
            # tweet_type: 6
            if idx == 6:
                if features[idx] == 'Retweet':
                    features[idx] = 0
                elif features[idx] == 'TopLevel':
                    features[idx] = 1
                elif features[idx] == 'Quote':
                    features[idx] = 2
                elif features[idx] == 'Reply':
                    features[idx] = 3
                else:
                    features[idx] = 4
                new_X.append(features[idx])
            # language: 7
            if idx == 7:
                langs = [
                    'D3164C7FBCF2565DDF915B1B3AEFB1DC', 
                    '22C448FF81263D4BAF2A176145EE9EAD', 
                    '06D61DCBBE938971E1EA0C38BD9B5446', 
                    'ECED8A16BE2A5E8871FD55F4842F16B1', 
                    'B9175601E87101A984A50F8A62A1C374', 
                    '4DC22C3F31C5C43721E6B5815A595ED6', 
                    '167115458A0DBDFF7E9C0C53A83BAC9B', 
                    '022EC308651FACB02794A8147AEE1B78', 
                    'FA3F382BC409C271E3D6EAF8BE4648DD', 
                    '125C57F4FA6D4E110983FB11B52EFD4E', 
                    'others']
                if features[idx] not in langs:
                    features[idx] = 10
                else:
                    features[idx] = langs.index(features[idx])

                new_X.append(features[idx])

            # ['engaged_with_user_is_verified', 'enaging_user_is_verified', 'engagee_follows_engager']
            # 12, 17, 19
            # if idx in [12, 17, 19]:
            if idx == 12 or idx == 17 or idx == 19:

                if features[idx] == 'true':
                    features[idx] = 1
                elif features[idx] == 'false':
                    features[idx] = 0

                new_X.append(features[idx])

            # numeric_df = 'engaged_with_user_follower_count', 
            # 'engaged_with_user_following_count',
            # 'enaging_user_follower_count', 
            # 'enaging_user_following_count'
            if idx == 10 or idx == 11 or idx == 15 or idx == 16:
                new_X.append(features[idx])

            # 'tweet_timestamp': 8
            if idx == 8:
                timestamp = int(features[idx])
                dt_object = datetime.fromtimestamp(timestamp)
                new_X.append(dt_object.year)
                new_X.append(dt_object.month)
                new_X.append(dt_object.day)
                new_X.append(dt_object.hour)
                new_X.append(dt_object.weekday())
            # 'engaged_with_user_account_creation': 13, 'enaging_user_account_creation': 18
            if idx == 13 or idx == 18:
                timestamp = int(features[idx])
                dt_object = datetime.fromtimestamp(timestamp)
                new_X.append(dt_object.year)
            
        X.append(new_X)
        # print(len(new_X))

        new_y = []
        for label, idx in labels_to_idx.items():
            # print("label {} has value {}".format(label, features[idx]))
            new_y.append(features[idx])
        y.append(new_y)

        # count += 1
        # if count == 5: break


    X = np.asarray(X)
    print(X)
    print(X.shape)
    # print(y)
    y = np.asarray(y)

    z = y[:,0].tolist()
    train(X,z,"model0_time")

    z = y[:,1].tolist()
    train(X,z,"model1_time")

    z = y[:,2].tolist()
    train(X,z,"model2_time")

    z = y[:,3].tolist()
    train(X,z,"model3_time")


