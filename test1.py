# from joblib import dump, load
import pandas as pd
import keras
import pickle
import xgboost as xgb
import numpy as np

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

            # 'present_links': 4, 'present_domains': 5
            if idx == 4 or idx == 5:

                if features[idx] != '':
                    features[idx] = 1
                else:
                    features[idx] = 0
                    
                new_X.append(features[idx])

            # present_media
            if idx == 3:
                if features[idx] == 'Video':
                    features[idx] = 3
                elif features[idx] == 'Photo':
                    features[idx] = 2
                elif features[idx] == 'Gif':
                    features[idx] = 1
                elif features[idx] == '':
                    features[idx] = 0
                else:
                    features[idx] = 4
                new_X.append(features[idx])

            # not use language, not use tweet type right
            if idx != 0 and idx != 1 and idx != 2 and idx != 7 and idx != 9 and idx != 14:
                # print("feature {} has value {}".format(feature, features[idx]))
                
                if features[idx] == 'true':
                    features[idx] = 1
                elif features[idx] == 'false':
                    features[idx] = 0

                if features[idx] == 'Retweet':
                    features[idx] = 1
                elif features[idx] == 'TopLevel':
                    features[idx] = 0
                elif features[idx] == 'Quote':
                    features[idx] = 2

                new_X.append(features[idx])

            
        X.append(new_X)


        # count += 1
        # if count == 5: break
    X = np.asarray(X)


with open('xgb_model', 'rb') as fd:
    clf = pickle.load(fd)

with open('xgb_model2', 'rb') as fd:
    clf2 = pickle.load(fd)

with open('xgb_model3', 'rb') as fd:
    clf3 = pickle.load(fd)

with open('xgb_model4', 'rb') as fd:
    clf4 = pickle.load(fd)



prediction = clf.predict(X)
prediction2 = clf2.predict(X)
prediction3 = clf3.predict(X)
prediction4 = clf4.predict(X)


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


