import numpy as np

all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
# print(all_features_to_idx)
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}

threshould = 10000

with open("val.tsv", encoding="utf-8") as f:

    # count = 0
    X = []
    F = []

    for line in f:

        # one line one data
        line = line.strip()
        # features is a list
        features = line.split("\x01")

        for feature, idx in all_features_to_idx.items():
            # print("feature {} has value {}".format(feature, features[idx]))
            '''
            if idx == 0:
                # print("feature {} has value {}".format(feature, features[idx]))
                split_index = features[idx].split()
                # using map() to perform conversion from str to int
                split_index = list(map(int, split_index))
                X.append(len(split_index))
            '''

            if idx == 10: # or idx == 15
                if int(features[idx]) > threshould and int(features[15]) > threshould:
                    F.append(1)
                else:
                    F.append(0)


        # count += 1
        # if count == 5: break

# print(X)
# np.savetxt('features/text_len_val.csv', X, delimiter=',', fmt='%i')
np.savetxt('features/bothAreFamous_val.csv', F, delimiter=',', fmt='%i')