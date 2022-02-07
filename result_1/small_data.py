import numpy as np
import os
import csv
from multiprocessing import Pool
import sys

# 用feature_out的tweet_id，把其他feature抓出來，轉移成一個較小的data_set(11999000筆)

with open("bert_feature.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data)

tweet_ids = data[:,0]
print(tweet_ids)
tweet_ids_v2 = []
tweet_ids_v2.append(tweet_ids[0:1500000])
tweet_ids_v2.append(tweet_ids[1500000:3000000])
tweet_ids_v2.append(tweet_ids[3000000:4500000])
tweet_ids_v2.append(tweet_ids[4500000:6000000])
tweet_ids_v2.append(tweet_ids[6000000:7500000])
tweet_ids_v2.append(tweet_ids[7500000:9000000])
tweet_ids_v2.append(tweet_ids[9000000:10500000])
tweet_ids_v2.append(tweet_ids[10500000:12000000])

bert_features = data[:,1:]
# (12000000,4)
# print(tweet_ids)   

all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
print(all_features_to_idx)

labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}


dirs = os.listdir("../train_part/")


selected_data = []

with open("../train_part/train.partaa", encoding="utf-8") as f:

    p = Pool(8)
    

    linenum = 0
    for line in f:
        linenum = linenum+1
        print("processing linenum:", linenum)

        # one line one data
        line = line.strip()
        # features is a list
        features = line.split("\x01")

        for feature, idx in all_features_to_idx.items():
            # print("feature {} has value {}".format(feature, features[idx]))
            # tweet_id
            if idx == 2:
                feat = features[idx]
                def f2(t_ids):
                    return feat in t_ids
                result_list = p.map(f2, tweet_ids_v2[:])
                #print("result_list: ", result_list)
                #sys.exit()
                if True in result_list:
                    selected_data.append(features)

                #if features[idx] in tweet_ids:
                #    selected_data.append(features)

        # count += 1
        # if count == 1000: break
print(len(selected_data))
# (12000000,20)

selected_data = np.asarray(selected_data)
# output = np.concatenate((selected_data, bert_features), axis=1)
np.savetxt('selected_aa.csv', selected_data, delimiter=',', fmt='%s')


