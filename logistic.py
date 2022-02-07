import os
import numpy as np
import pickle
import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))

labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}


path = "train_part/"
dirs = os.listdir( path )

for file in dirs:

    X = []
    y = []

    with open("train_part/"+file, encoding="utf-8") as f:

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
            # if count == 1000: break


    y = np.asarray(y)

    # y1 = y[:,0].tolist()
    y2 = y[:,1].tolist()
    y3 = y[:,2].tolist()
    y4 = y[:,3].tolist()

    # for i in range(len(y1)):
    #     if y1[i] == '':
    #         y1[i] = 0
    #     else:
    #         y1[i] = 1
    
    for i in range(len(y2)):
        if y2[i] == '':
            y2[i] = 0
        else:
            y2[i] = 1

    for i in range(len(y3)):
        if y3[i] == '':
            y3[i] = 0
        else:
            y3[i] = 1

    for i in range(len(y4)):
        if y4[i] == '':
            y4[i] = 0
        else:
            y4[i] = 1


    print("padding")
    X = keras.preprocessing.sequence.pad_sequences(X, maxlen=200, padding='post')
    print("splitting data")

    # X_train, X_val, y_train, y_val = train_test_split(X, y1, test_size=0.3, random_state=42)
    X_train2, X_val2, y_train2, y_val2 = train_test_split(X, y2, test_size=0.3, random_state=42)
    X_train3, X_val3, y_train3, y_val3 = train_test_split(X, y3, test_size=0.3, random_state=42)
    X_train4, X_val4, y_train4, y_val4 = train_test_split(X, y4, test_size=0.3, random_state=42)
    # print(X_train, y_train)
    print("start learning")

    # clf = LogisticRegression(warm_start=True, max_iter=1000).fit(X_train, y_train)
    clf2 = LogisticRegression(warm_start=True, max_iter=1000).fit(X_train2, y_train2)
    clf3 = LogisticRegression(warm_start=True, max_iter=1000).fit(X_train3, y_train3)
    clf4 = LogisticRegression(warm_start=True, max_iter=1000).fit(X_train4, y_train4)
    # print(clf.score(X_train, y_train))
    # print(clf.score(X_val, y_val))

    # 0.9743814285714286
    # 0.9741925

    # 0.9743389285714286
    # 0.9743711111111111

    # 0.9742775
    # 0.9743366666666666

    # 0.9742604761904762
    # 0.9742775

    # 0.9742558333333333
    # 0.9742577777777778

    # 0.9742971428571429
    # 0.9742322222222223

    # 0.9742497619047619
    # 0.9742830555555556

    # 0.9742607142857143
    # 0.9742502777777777

    # 0.9741534952094723
    # 0.9745180076726397

    print(clf2.score(X_train2, y_train2))
    print(clf2.score(X_val2, y_val2))

    print(clf3.score(X_train3, y_train3))
    print(clf3.score(X_val3, y_val3))

    print(clf4.score(X_train4, y_train4))
    print(clf4.score(X_val4, y_val4))

# print(clf.predict(X_val))
# print(clf.predict_proba(X[:2, :]))

# print(clf.score(X_train, y_train))
# print(clf.score(X_val, y_val))

print(clf2.score(X_train2, y_train2))
print(clf2.score(X_val2, y_val2))

print(clf3.score(X_train3, y_train3))
print(clf3.score(X_val3, y_val3))

print(clf4.score(X_train4, y_train4))
print(clf4.score(X_val4, y_val4))

# dump(clf, 'model.joblib')
dump(clf2, 'model2.joblib') 
dump(clf3, 'model3.joblib') 
dump(clf4, 'model4.joblib')  

