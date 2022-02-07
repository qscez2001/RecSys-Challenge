#!/bin/sh



for entry in "train_part"/*;
do
    echo "$entry"
    if grep -q "60968762145D2AF58A58AFB376B2B00C" 
    then
        echo "1"
        # code if found
    else
        # code if not found
        echo "0"
    fi
done