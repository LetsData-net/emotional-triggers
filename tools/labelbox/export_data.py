import os
import json

import hashlib

from labelbox import Client

LABELBOX_API_KEY = os.environ['LABELBOX_API_KEY']
LABELBOX_PROJECT_NAME = os.environ['LABELBOX_PROJECT_NAME']

DATA_DIR = 'data'


def get_hash(item, n_hash=8) -> str:
    hash_object = hashlib.md5(str(item).encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig[:n_hash]


def export_raw_annotations():
    client = Client(api_key=LABELBOX_API_KEY)
    project = client.get_project(LABELBOX_PROJECT_NAME)

    export_task = project.export_v2(params={
        "data_row_details": True,
        "project_details": True,
        "label_details": True,
        "interpolated_frames": True
    })

    export_task.wait_till_done()
    if export_task.errors:
        print(export_task.errors)

    raw_annotations = export_task.result
    print(f'Loaded {len(raw_annotations)} annotations')

    return raw_annotations


def process_label(label) -> dict:
    created_by = get_hash(label['label_details']['created_by'], n_hash=8)
    created_at = label['label_details']['created_at']

    emotion_label = None
    emotion_class_label = None
    trigger_level_label = None
    emotional_triggering_phrases = []
    emotional_triggering_phrases_locations = []
    description = None

    for classification in label['annotations']['classifications']:

        if classification['value'] == 'emotion':
            emotion_label = classification['checklist_answers'][0]['value']

        elif classification['value'] == 'emotion_class':
            emotion_class_label = classification['radio_answer']['value']

        elif classification['value'] == 'trigger_level_1_low_5_high':
            trigger_level_label = classification['checklist_answers'][0]['value']

        elif classification['value'] == 'optional_description':
            description = classification['text_answer']['content']

    for obj in label['annotations']['objects']:

        if obj['value'] == 'emotional_triggering_phrases':
            emotional_triggering_phrases.append(obj['location']['token'])
            emotional_triggering_phrases_locations.append((obj['location']['start'], obj['location']['end']))

    if emotion_label is None:
        return {}

    item_label = {
        "emotion": emotion_label,
        "emotion_class": emotion_class_label,
        "trigger_level": trigger_level_label,
        "context": emotional_triggering_phrases,
        "context_locations": emotional_triggering_phrases_locations,
        "description": description,
        "created_at": created_at,
        "created_by": created_by,
    }

    print(item_label)

    return item_label


def main():

    annotations_dict = dict()

    for annotation in export_raw_annotations():

        for label in annotation["projects"].get(LABELBOX_PROJECT_NAME, {}).get("labels", []):

            processed_label = process_label(label)
            processed_label["text"] = annotation["data_row"].get("row_data")
            processed_label["id"] = annotation["data_row"].get("global_key")

            created_by = processed_label["created_by"]

            if created_by not in annotations_dict:
                annotations_dict[created_by] = []

            annotations_dict[created_by].append(processed_label)

    for k, v in annotations_dict.items():
        print(k, len(v))

    with open(f"{DATA_DIR}/labelbox_sample_annotated.json", "w") as out_file:
        json.dump(annotations_dict, out_file, ensure_ascii=False)


if __name__ == '__main__':
    main()
