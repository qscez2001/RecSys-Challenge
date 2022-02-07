# from joblib import dump, load
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
from datetime import datetime
import csv

all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))

labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}


X = []

tweet_id = []
engaged_with_user_id = []

print("preprocessing...")

with open("features/text_len_val.csv", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    text_len = np.array(data)

with open("features/bothAreFamous_val.csv", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)

    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    bothAreFamous = np.array(data)

with open("val.tsv", encoding="utf-8") as f:

    # count = 0

    for line in f:

        # one line one data
        line = line.strip()
        # features is a list
        features = line.split("\x01")

        new_X = []
        for feature, idx in all_features_to_idx.items():
            # print("feature {} has value {}".format(feature, features[idx]))

            if idx == 2:
                tweet_id.append(features[idx])
            if idx == 14:
                engaged_with_user_id.append(features[idx])

            # 'present_links': 4 present_domains: 5
            if idx == 4 or idx == 5:

                if features[idx] != '':
                    features[idx] = 1
                else:
                    features[idx] = 0
                    
                new_X.append(features[idx])
            # 'present_media': 3
            if idx == 3:
                # if features[idx] == 'Photo':
                #     features[idx] = 0
                # elif features[idx] == 'Video':
                #     features[idx] = 1
                # elif features[idx] == 'Gif':
                #     features[idx] = 2
                # else:
                #     features[idx] = 3

                # new_X.append(features[idx])
                if features[idx] == '':
                    media_count = [0, 0, 0]
                else:
                    token = features[idx].split('\t')
                    p_num = token.count('Photo')
                    v_num = token.count('Video')
                    g_num = token.count('Gif')
                    media_count = [p_num, v_num, g_num]   

                for count in media_count:
                    new_X.append(count)
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
                langs = ['D3164C7FBCF2565DDF915B1B3AEFB1DC', '22C448FF81263D4BAF2A176145EE9EAD', '06D61DCBBE938971E1EA0C38BD9B5446', 'ECED8A16BE2A5E8871FD55F4842F16B1', 'B9175601E87101A984A50F8A62A1C374', '4DC22C3F31C5C43721E6B5815A595ED6', '167115458A0DBDFF7E9C0C53A83BAC9B', '022EC308651FACB02794A8147AEE1B78', 'FA3F382BC409C271E3D6EAF8BE4648DD', '125C57F4FA6D4E110983FB11B52EFD4E', 'others']
                if features[idx] not in langs:
                    features[idx] = 10
                else:
                    features[idx] = langs.index(features[idx])

                new_X.append(features[idx])

            # ['engaged_with_user_is_verified', 'enaging_user_is_verified', 'engagee_follows_engager']
            # 12, 17, 19
            if idx == 12 or idx == 17 or idx == 19:
                # print("feature {} has value {}".format(feature, features[idx]))
                if features[idx] == 'true':
                    features[idx] = 1
                elif features[idx] == 'false':
                    features[idx] = 0

                new_X.append(features[idx])

            # numeric_df = df[['engaged_with_user_follower_count', 'engaged_with_user_following_count', 'enaging_user_follower_count', 'enaging_user_following_count']]
            # 'tweet_timestamp': 8,
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


        # count += 1
        # if count == 5: break
    X = np.asarray(X)

    X = np.concatenate((X, text_len), axis=1)
    print(X.shape)

    X = np.concatenate((X, bothAreFamous), axis=1)
    print(X.shape)

print("predicting...")

with open('model0_onehot', 'rb') as fd:
    clf = pickle.load(fd)

with open('model1_onehot', 'rb') as fd:
    clf2 = pickle.load(fd)

with open('model2_onehot', 'rb') as fd:
    clf3 = pickle.load(fd)

with open('model3_onehot', 'rb') as fd:
    clf4 = pickle.load(fd)



prediction = clf.predict(X)
prediction2 = clf2.predict(X)
prediction3 = clf3.predict(X)
prediction4 = clf4.predict(X)

'''
[[0.9009259  0.09907416]
 [0.97348577 0.02651422]
 [0.9573058  0.04269422]
 [0.97348577 0.02651422]
 [0.95879745 0.04120255]]
 '''
# prediction = clf.predict_proba(X)
# # print(prediction)
# prediction2 = clf2.predict_proba(X)
# prediction3 = clf3.predict_proba(X)
# prediction4 = clf4.predict_proba(X)


'''
{'text_ tokens': 0, 'hashtags': 1, 'tweet_id': 2, 'present_media': 3, 'present_links': 4, 
'present_domains': 5, 'tweet_type': 6, 'language': 7, 'tweet_timestamp': 8, 
'engaged_with_user_id': 9, 'engaged_with_user_follower_count': 10, 
'engaged_with_user_following_count': 11, 'engaged_with_user_is_verified': 12, 
'engaged_with_user_account_creation': 13, 'enaging_user_id': 14, 
'enaging_user_follower_count': 15, 'enaging_user_following_count': 16, 
'enaging_user_is_verified': 17, 'enaging_user_account_creation': 18, 
'engagee_follows_engager': 19}

'tweet_id': 2
'engaged_with_user_id': 9
'enaging_user_id': 14

Please upload predictions for each engagement type in csv format where each line 
consists of: <Tweet_Id>,<User_Id>,<Prediction>

'''
# {"reply_timestamp": 20, "retweet_timestamp": 21, 
# "retweet_with_comment_timestamp": 22, "like_timestamp": 23}

print("saving...")

d = {'col1': tweet_id, 'col2': engaged_with_user_id, 'col3': prediction}
df = pd.DataFrame(data=d)
df.to_csv('reply.csv', index=False, header=False)

d = {'col1': tweet_id, 'col2': engaged_with_user_id, 'col3': prediction2}
df = pd.DataFrame(data=d)
df.to_csv('retweet.csv', index=False, header=False)

d = {'col1': tweet_id, 'col2': engaged_with_user_id, 'col3': prediction3}
df = pd.DataFrame(data=d)
df.to_csv('retweet_with_comment.csv', index=False, header=False)

d = {'col1': tweet_id, 'col2': engaged_with_user_id, 'col3': prediction4}
df = pd.DataFrame(data=d)
df.to_csv('like.csv', index=False, header=False)


