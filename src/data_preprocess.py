import random
import os
import json

FILE_PATH = "./tmp"

def preprocess(context, USER, AI ):

    data = []
    
    full_path = os.path.join(FILE_PATH,"syntheic_data.json" )
    with open(full_path, 'r', encoding="utf-8") as f:
            data.append(json.load(f))

    ds_formatted = [
        {"messages": [
            {"role": "system", "content": context},
            {'role': 'user', 'content': i[USER]},
            {'role': 'assistant', 'content': i[AI]}]} for i in data[0]
    ]

    random.shuffle(ds_formatted)

    # Calculate the number of samples in ds_formatted
    total_samples = len(ds_formatted)

    # Calculate 70% of the total for training
    train_size = int(0.8 * total_samples)

    # Splitting data for train 
    ds_train = ds_formatted[:train_size]
    ds_test = ds_formatted[train_size:]

    return ds_train, ds_test
