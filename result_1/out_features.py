import numpy as np
import os
import json



# 所以有一個 json 負責記錄每個檔案裏的推文 ID。

files = os.listdir("train_result")

# parse names e.g. xzfsb
names = []
for f in files:
    name = f.replace("model_output_", "")
    name = name.replace(".npy", "")
    names.append(name)

tweet_ids = []
# load tweet_id
with open('predict.json') as f:
    data = json.load(f)

    # print(len(data.keys()))
    for name in names:
        if name in data:
            tweet_ids.append(data[name])
        else:
            print(name)
        

tweet_ids = np.asarray(tweet_ids)
tweet_ids = np.reshape(tweet_ids, (-1,1))
print(tweet_ids.shape)


# 先concat成 (12000000, 5)


# load probability score
flag = 0
for file in files:
    if flag == 0:
        a = np.load("train_result/"+file)
        flag = 1
    else:

        a = np.concatenate((a, np.load("train_result/"+file)), axis=0)
        # print(a.shape)

print(a.shape)
# (12000000, 4)
features = np.concatenate((tweet_ids, a), axis=1)
print(features.shape)

# tweet_id
np.savetxt('bert_feature.csv', features, delimiter=',', fmt='%s')

