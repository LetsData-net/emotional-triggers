import os

import pandas as pd

from tqdm import tqdm
from labelbox import Client

LABELBOX_API_KEY = os.environ['LABELBOX_API_KEY']
LABELBOX_DATASET_NAME = os.environ['LABELBOX_DATASET_NAME']

DATA_DIR = 'data'


if __name__ == '__main__':
    input_data = pd.read_csv(f'{DATA_DIR}/labelbox_sample.csv')

    client = Client(api_key=LABELBOX_API_KEY)
    dataset = client.create_dataset(name=LABELBOX_DATASET_NAME)

    for i, row in tqdm(input_data.iterrows(), total=len(input_data)):
        try:
            dataset.create_data_row(row_data=row["text"], global_key=f'{row["id"]}')
        except Exception as ex:
            print(f'Failed to create data row for {i}-th item: {row}. Reason: {ex}')
