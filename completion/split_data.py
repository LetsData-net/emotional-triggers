import json
import os.path

import numpy as np
import pandas as pd

FT_TRAIN_NUM = 50
FT_VAL_NUM = 20

np.random.seed(42)

DATA_DIR = 'data'

annotator2persona_mapping = {"persona_1": "d16ec044", "persona_2": "3fefb024"}


def dump_split_data_to_csv(data_dict: dict) -> None:
    output_file_name = f"{DATA_DIR}/labelbox_sample_w_split.csv"
    if os.path.exists(output_file_name):
        return

    data_dict_t = dict()
    for split_name, ids in data_dict.items():
        for text_id in ids:
            data_dict_t[text_id] = split_name

    data = pd.read_csv(f"{DATA_DIR}/labelbox_sample.csv")
    data["split_name"] = data["id"].map(data_dict_t)
    data.to_csv(output_file_name)


def get_or_load_data_split(publications_file_name: str) -> dict:
    output_file_name = f"{DATA_DIR}/data_split_ids.json"

    if os.path.exists(output_file_name):
        data_dict = json.load(open(output_file_name))

        dump_split_data_to_csv(data_dict)

        return data_dict

    data = pd.read_csv(publications_file_name)
    data = data.sort_values(by="date").reset_index(drop=True)

    ft_training = data.iloc[:FT_TRAIN_NUM, :]["id"].tolist()
    ft_validation = data.iloc[FT_TRAIN_NUM:FT_TRAIN_NUM + FT_VAL_NUM, :]["id"].tolist()
    evaluation = data.iloc[FT_TRAIN_NUM + FT_VAL_NUM:, :]["id"].tolist()

    print(f"Selected {len(ft_training)} samples for FT training")
    print(f"Selected {len(ft_validation)} samples for FT validation")
    print(f"Selected {len(evaluation)} samples for evaluation")

    data_dict = {
        "ft_training": ft_training, "ft_validation": ft_validation, "evaluation": evaluation,
    }

    with open(output_file_name, "w") as o_file:
        json.dump(data_dict, o_file)

    return data_dict


def enrich_data(annotated_data: pd.DataFrame, created_by: str) -> pd.DataFrame:
    descriptions_file = f"{DATA_DIR}/{created_by}_descriptions.csv"
    if not os.path.exists(descriptions_file):
        return annotated_data
    descriptions = pd.read_csv(descriptions_file)[["id", "description"]]
    descriptions = descriptions[~descriptions["description"].isna()]
    descriptions = descriptions.drop_duplicates("id").reset_index(drop=True)
    annotated_data = annotated_data.merge(descriptions, on="id", how="left", suffixes=("_x", "_y"))
    annotated_data["description"] = np.where(~annotated_data["description_y"].isna(), annotated_data["description_y"],
                                             annotated_data["description_x"])
    annotated_data = annotated_data.drop(columns=["description_x", "description_y"])
    return annotated_data


def main():

    data_split_dict = get_or_load_data_split(f"{DATA_DIR}/labelbox_sample.csv")

    data_annotated = pd.read_json(f"{DATA_DIR}/labelbox_sample_annotated.json")

    for created_by in data_annotated.columns:

        annotator_data = pd.json_normalize(data_annotated[created_by])
        annotator_data = enrich_data(annotator_data, created_by)

        for split_name, split_ids in data_split_dict.items():

            annotator_data_split = annotator_data[annotator_data["id"].isin(split_ids)]

            if len(annotator_data_split) != len(split_ids):
                missing_ids = set(split_ids).difference(annotator_data_split["id"])
                raise ValueError(f"Annotator {created_by} is missing data for ids: {missing_ids}")

            print(f"Number of descriptions is {len(annotator_data_split)-annotator_data_split.description.isna().sum()}")

            print(f"Saving {len((annotator_data_split))} for split: {split_name} for annotator: {created_by}.")
            annotator_data_split.to_csv(f"{DATA_DIR}/train_test_eval_split/{created_by}_{split_name}.csv", index=None)


if __name__ == '__main__':
    main()
