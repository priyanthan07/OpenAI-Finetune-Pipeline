import os
import io
import json
import openai
from dotenv import load_dotenv
from openai import OpenAI, File as OpenAIFile

load_dotenv()


TMP_DIR_PATH = './tmp'

client = OpenAI()

def upload( train_data, test_data):

    os.makedirs(TMP_DIR_PATH, exist_ok=True)

    train_data_path = f'{TMP_DIR_PATH}/_train.jsonl'
    test_data_path = f'{TMP_DIR_PATH}/_test.jsonl'

    with open(train_data_path, 'w') as file:
        for line in train_data:
            json.dump(line, file)
            file.write("\n")

    with open(test_data_path, 'w') as file:
        for line in test_data:
            json.dump(line, file)
            file.write("\n")

    with open(train_data_path, "rb") as train_file:
        train = openai.files.create(
            file=train_file,
            purpose="fine-tune"
        )

    train_id = train.id

    with open(test_data_path, "rb") as test_file:
        test = client.files.create(
            file=test_file,
            purpose="fine-tune"
        )

    test_id = test.id

    # # delete those 2 files
    os.remove(train_data_path)
    os.remove(test_data_path)
    
    return train_id, test_id
    