import numpy as np
import os
import csv
from multiprocessing import Pool
import time

start_time = time.time()
# 用feature_out的tweet_id，把其他feature抓出來，轉移成一個較小的data_set(11999000筆)

with open("bert_feature.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data)

tweet_ids = data[:,0]
# bert_features = data[:,1:]
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


def f(file):
    selected_data = []

    with open("../train_part/"+file, encoding="utf-8") as f:

        # count = 0

        linenum = 0
        for line in f:
            print("processing linenum ", linenum)
            linenum = linenum+1

            # one line one data
            line = line.strip()
            # features is a list
            features = line.split("\x01")

            for feature, idx in all_features_to_idx.items():
                # print("feature {} has value {}".format(feature, features[idx]))
                # tweet_id
                if idx == 2:
                    if features[idx] in tweet_ids:
                        selected_data.append(features)

            # count += 1
            # if count == 1000: break
    print(len(selected_data))
    # (12000000,20)

    selected_data = np.asarray(selected_data)
    # output = np.concatenate((selected_data, bert_features), axis=1)
    np.savetxt('select_{}.csv'.format(file), selected_data, delimiter=',', fmt='%s')


if __name__ == '__main__':
    p = Pool(6)
    # p.map(f, dirs[:1])
    p.map(f, dirs[1:2])
    print("ddd")
    print("--- %s seconds ---" % (time.time() - start_time))


