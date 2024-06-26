import ast
import json

import tiktoken

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

PERSONAS_LIST = ["persona_1", "persona_2"]
RANDOM_SEEDS = [7, 33, 42]
APPROACHES_LIST = ["no_context", "context_zero_shot", "context_few_shot_5", "context_few_shot_10", "context_fine_tuning"]

DATA_DIR = 'data'

annotator2persona_mapping = {"persona_1": "d16ec044", "persona_2": "3fefb024", "assistant": "assistant"}

encoding = tiktoken.get_encoding("cl100k_base")


def evaluate_classification(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return {"precision": precision, "recall": recall, "f1_score": f1}


def get_classification_report(df_true: pd.DataFrame, df_predicted: pd.DataFrame, column_name: str) -> dict:
    return evaluate_classification(true_labels=df_true[column_name].tolist(),
                                   predicted_labels=df_predicted[column_name].tolist())


def calc_strict_match(true_labels, predicted_labels, corrupted_labels: list[str] = None):
    correct_pairs = 0
    predicted_pairs = 0
    annotated_pairs = 0
    for true_context, predicted_context in zip(true_labels, predicted_labels):
        predicted_pairs += len(predicted_context)
        annotated_pairs += len(true_context)
        correct_pairs += len(list(set(true_context).intersection(predicted_context)))
        # take into account incorrect spans generated
        if corrupted_labels is not None:
            predicted_pairs += len(corrupted_labels)
    precision = correct_pairs / predicted_pairs
    recall = correct_pairs / annotated_pairs
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1-score": f1}


def calc_span_intersection(true_context: list[tuple], predicted_context: list[tuple]):
    correct_pairs = []
    for true_start, true_end in true_context:
        true_chars = list(range(true_start, true_end + 1))
        for pred_start, pred_end in predicted_context:
            pred_chars = list(range(pred_start, pred_end + 1))
            intersection = sorted(set(true_chars).intersection(set(pred_chars)))
            if not intersection:
                continue
            correct_pairs.append((intersection[0], intersection[-1]))
    return correct_pairs


def calc_proportional_match(true_labels, predicted_labels, corrupted_labels):
    correct_pairs = 0
    predicted_pairs = 0
    annotated_pairs = 0
    for true_context, predicted_context in zip(true_labels, predicted_labels):
        annotated_pairs += sum([e - s + 1 for s, e in true_context])
        predicted_pairs += sum([e - s + 1 for s, e in predicted_context])
        correct_pairs += sum([e - s + 1 for s, e in calc_span_intersection(true_context, predicted_context)])
        # take into account incorrect spans generated
        if corrupted_labels is not None:
            predicted_pairs += sum([len(lab) for lab in corrupted_labels])
    precision = correct_pairs / predicted_pairs
    recall = correct_pairs / annotated_pairs
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1-score": f1}


def get_spans_detection_report(df_true: pd.DataFrame, df_predicted: pd.DataFrame, column_name: str) -> dict:
    if column_name == 'strict':
        report = calc_strict_match(true_labels=df_true["context"].tolist(),
                                   predicted_labels=df_predicted["context"].tolist(),
                                   corrupted_labels=df_predicted["context_corrupted"].tolist())
    elif column_name == 'proportional':
        report = calc_proportional_match(true_labels=df_true["context_locations"].tolist(),
                                         predicted_labels=df_predicted["context_locations"].tolist(),
                                         corrupted_labels=df_predicted["context_corrupted"].tolist())
    else:
        raise ValueError(column_name)
    return report


def compare(df_true, df_predicted, column_name):
    df_compare = df_true[["id", "text", column_name]]
    df_compare = df_compare.merge(df_predicted[["id", column_name]], on="id", suffixes=("_true", "_predicted"))
    df_compare = df_compare[df_compare[f"{column_name}_true"] != df_compare[f"{column_name}_predicted"]]
    df_compare.to_csv(f'compare_{column_name}.csv', index=None)


def get_report(df_true: pd.DataFrame, df_predicted: pd.DataFrame, column_name: str, overlap_type: str = None):
    print("Column:", column_name)
    if column_name in ["context", "context_locations"]:
        report = get_spans_detection_report(df_true, df_predicted, overlap_type)
        return {f'{column_name}_{overlap_type}_{k}': v for k, v in report.items()}
    elif column_name in ["emotion", "emotion_class", "trigger_level"]:
        report = get_classification_report(df_true, df_predicted, column_name)
        return {f'{column_name}_{k}': v for k, v in report.items()}
        # compare(df_true, df_predicted, column_name)
    else:
        raise ValueError(column_name)


def load_true(file_name: str) -> pd.DataFrame:
    true_df = pd.read_csv(file_name).sort_values(by='id')
    true_df["context"] = true_df["context"].apply(ast.literal_eval)
    true_df["context_locations"] = true_df["context_locations"].apply(ast.literal_eval)
    return true_df


def process_and_validate_context(data: pd.DataFrame) -> pd.DataFrame:
    data_rows = data.to_dict("records")
    for data_row in data_rows:
        context_locations = []
        corrupted_context = []
        context_list = data_row["context"]
        if not isinstance(context_list, list):
            if context_list in [np.nan, None]:
                context_list = []
            else:
                try:
                    context_list = ast.literal_eval(context_list)
                except Exception as ex:
                    print(f"{context_list}: {ex}")
                    corrupted_context += [context_list]
                    context_list = []
        for context_item in context_list:
            if isinstance(context_item, list):
                corrupted_context.append(context_item)
                continue
            start = data_row["text"].lower().find(context_item.lower())
            if start != -1:
                context_locations.append((start, start+len(context_item)))
            else:
                corrupted_context.append(context_item)
                print(f"Skipping corrupted context {context_item}")
        data_row["context_locations"] = context_locations
        data_row["context_corrupted"] = corrupted_context
        data_row["context"] = [c for c in context_list if c not in corrupted_context]
    processed_data = pd.DataFrame(data_rows)
    return processed_data


def load_predicted(persona_name, approach, seed) -> pd.DataFrame:
    predictions = list()

    if approach == "no_context":
        persona_name = "assistant"

    file_name = f"{DATA_DIR}/completed/{persona_name}_{approach}_seed={seed}.jsonl"
    with open(file_name) as input_file:
        for line in input_file.readlines():
            predictions.append(json.loads(line))

    df_predicted = pd.DataFrame(predictions)
    print(df_predicted.shape)

    df_predicted_output = pd.json_normalize(df_predicted["output"])
    df_predicted = pd.concat([df_predicted.drop(columns=["output"]), df_predicted_output], axis=1)
    print(df_predicted.shape)

    assert len(df_predicted) == 166

    texts = pd.read_csv(f"{DATA_DIR}/labelbox_sample.csv")[["id", "text"]]
    print(texts.shape)

    df_predicted = df_predicted.merge(texts, on="id", how="left")
    print(df_predicted.shape)

    assert df_predicted.text.isna().sum() == 0, "missing texts"

    df_predicted = process_and_validate_context(df_predicted)

    df_predicted['trigger_level'] = df_predicted['trigger_level'].apply(get_int)
    df_predicted['emotion_class'] = np.where(~df_predicted['emotion_class'].isin(['positive', 'negative']),
                                             'corrupted', df_predicted['emotion_class'])
    df_predicted['emotion'] = np.where(
        ~df_predicted['emotion'].isin(['anger', 'love', 'joy', 'surprise', 'fear', 'sadness', 'undefined']
                                      ), 'corrupted', df_predicted['emotion'])

    df_predicted = df_predicted.sort_values(by='id').reset_index(drop=True)

    return df_predicted


def get_int(val):
    try:
        return int(val)
    except:
        return -1


def generate_beautiful_report(full_report, report_name="report"):
    df = []
    for persona_name, approaches in full_report.items():
        for approach, approach_vals in approaches.items():
            for column, column_vals in approach_vals.items():
                if column == 'failed':
                    df.append({
                        "persona_name": persona_name,
                        "approach_name": approach,
                        "name": column,
                        "metric": "count",
                    })
                    continue
                for cat, cat_dict in column_vals.items():
                    item = {
                        "persona_name": persona_name,
                        "approach_name": approach,
                        "name": column,
                        "metric": cat,
                    }
                    if not isinstance(cat_dict, dict):
                        cat_dict = {cat: cat_dict}
                        print(cat_dict)
                    item.update(cat_dict)
                    df.append(item)
    df = pd.DataFrame.from_records(df)
    df.to_csv(f"{report_name}.csv", index=None)
    df_simple = df[df.metric.isin(["strict", "proportional",
                                   "weighted avg"])].drop(columns=["support", "accuracy", "persona_name"])
    df_simple = df_simple.groupby(["approach_name", "name", "metric"]).mean().reset_index()
    df_simple = df_simple.sort_values(["name", "metric", "approach_name"]
                                      )[["name", "metric", "approach_name", "precision", "recall", "f1-score"]]
    df_simple.to_csv(f"{report_name}_simple.csv", index=None)


def add_mean_std_row(df):
    # Check if DataFrame is empty
    if df.empty:
        return df

    # Calculate mean and std for all numerical columns
    stats = df.select_dtypes(include=[np.number]).agg(['mean', 'std'])

    # Calculate the mean ± 3*std for each numerical column
    result_row = stats.loc['mean'].round(3).astype(str) + '±' + (3 * stats.loc['std']).round(2).astype(str)

    # Append the new row to the DataFrame
    new_row = pd.DataFrame([result_row], columns=df.select_dtypes(include=[np.number]).columns)

    # Combine the new row with the original DataFrame
    result_df = pd.concat([df, new_row], ignore_index=True)

    # Fill non-numeric columns in the new row with NaN
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if col == 'persona_name':
            val = 'average'
        elif col == 'approach_name':
            val = result_df.approach_name.values[0]
        else:
            val = np.nan
        result_df.loc[result_df.index[-1], col] = val

    return result_df


def main():

    full_report = dict()

    for approach in APPROACHES_LIST:

        full_report[approach] = list()

        for persona_name in PERSONAS_LIST:

            df_true_ = load_true(
                f"{DATA_DIR}/train_test_eval_split/{annotator2persona_mapping[persona_name]}_evaluation.csv")

            for seed in RANDOM_SEEDS:

                df_true = df_true_.copy()

                df_predicted = load_predicted(persona_name, approach, seed)

                assert len(df_true) == len(df_predicted), "len mismatch"
                assert np.alltrue([i1 == i2 for i1, i2 in zip(df_true.id, df_predicted.id)]), "ids mismatch"

                failed_to_generate = df_predicted[(df_predicted["emotion"] == 'corrupted') |
                                                  (df_predicted["emotion_class"] == 'corrupted') |
                                                  (df_predicted["trigger_level"] == -1)]["id"].tolist()

                emotion_report = get_report(df_true, df_predicted, "emotion")
                emotion_class_report = get_report(df_true, df_predicted, "emotion_class")
                trigger_level_report = get_report(df_true, df_predicted, "trigger_level")
                strict_context_report = get_report(df_true, df_predicted, "context", "strict")
                proportional_context_report = get_report(df_true, df_predicted, "context", "proportional")

                item = {
                    'failed': len(failed_to_generate)/len(df_true),
                    'seed': seed,
                    'persona_name': persona_name,
                    'approach_name': approach,
                }
                item.update(emotion_report)
                item.update(emotion_class_report)
                item.update(trigger_level_report)
                item.update(trigger_level_report)
                item.update(strict_context_report)
                item.update(proportional_context_report)

                full_report[approach].append(item)

        approach_report = pd.DataFrame(full_report[approach])
        approach_report = add_mean_std_row(approach_report)
        full_report[approach] = approach_report

    df = pd.concat(full_report.values()).drop(columns=['seed'])
    df = df[df.persona_name == 'average']
    df.to_csv('report_all.csv', index=None)
    print(df)


if __name__ == '__main__':
    main()
