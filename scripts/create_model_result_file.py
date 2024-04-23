# calculates the race results for a set of files. The code reads the input file, processes the header and lines, and writes the output file. The code is executed for each file in the file_list list. The input and output paths are hard-coded in the script.
from threshold_config import threshold_dict


def _calculate_scores(tp, fp, tn, fn):
    test_count = tp + fp + tn + fn

    if tp + fp == 0:
        precision = 'N/A'
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 'N/A'
    else:
        recall = tp / (tp + fn)

    if precision == 'N/A' or recall == 'N/A':
        f1_score = 'N/A'
    elif precision + recall == 0:
        f1_score = 'N/A'
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    if test_count == 0:
        accuracy = 'N/A'
    else:
        accuracy = (tp + tn) / test_count

    if tn + fp == 0:
        specificity = 'N/A'
    else:
        specificity = tn / (tn + fp)

    return f1_score, accuracy, recall, precision, specificity


def write_results_to_file(race, threshold, input_file, output_file):
    # Read input file
    with open(input_file, 'r') as f_in:
        header = f_in.readline()
        lines = f_in.readlines()

    tp, fp, tn, fn = 0, 0, 0, 0

    # Process lines
    count = 0
    for line in lines:
        parts = line.strip('/t').split()
        # print(parts)

        distance = float(parts[7][:7])
        print(distance)
        print(threshold)

        if parts[0] == parts[2] and distance < threshold:
            tp = tp + 1
        elif parts[0] == parts[2] and distance > threshold:
            fn = fn + 1
        elif parts[0] != parts[2] and distance < threshold:
            fp = fp + 1
        elif parts[0] != parts[2] and distance > threshold:
            tn = tn + 1

        count += 1
        if count > 100000:
            break

    # Calculate accuracy
    f1_score, accuracy, recall, precision, specificity = _calculate_scores(
        tp, fp, tn, fn)

    with open(output_file, 'w') as f_out:
        pass

    # Write output file
    with open(output_file, 'a') as f_out:
        f_out.write(race)
        f_out.write(
            f"\n\tTrue Positive:\t{tp}\n\tTrue Negative:\t{tn}\n\tFale  Positive:\t{fp}\n\tFalse Negative:\t{fn}\n")
        f_out.write(f"\tMatching Test Count:\t{tp + fn}\n")
        f_out.write(f"\tNon Matching Test Count:\t{tn + fp}\n")
        f_out.write(f"\tThreshold:\t{threshold}\n\n")

        f_out.write(
            f"\tf1 score:\t{f1_score}\n\taccuracy:\t{accuracy}\n\trecall:\t\t{recall}\n\tprecision:\t{precision}\n\tspecificity:\t{specificity}\n\n")


if __name__ == "__main__":

    race_list = ["African", "Asian", "Caucasian", "Indian"]
    # race_list = ["African"]
    model_list = ["DeepFace", "ArcFace", "Facenet", "Facenet512"]

    # NOTE: Model Thresholds are defined in threshold_config.py
    for model in model_list:
        path = "testing_results/verification/" + model + "/"
        output_file = path + "Results.txt"

        for race in race_list:
            input_file = path + race + "_results.txt"
            threshold = threshold_dict[model]
            write_results_to_file(race, threshold, input_file, output_file)
        print("Output file generated successfully.")
