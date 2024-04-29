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

    # mode = "Optimal"
    # threshold_dictionary = {
    #         "DeepFace": [0.35, 0.36, 0.3, 0.3], 
    #         "ArcFace": [0.68, 0.67, 0.73, 0.69],
    #         "Facenet": [0.47, 0.46, 0.61, 0.55],
    #         "Facenet512": [0.44, 0.44, 0.59, 0.5]
    #     }
    
    mode = "Standard"
    threshold_dictionary = {
            "DeepFace": [0.23, 0.23, 0.23, 0.23], 
            "ArcFace": [0.68, 0.68, 0.68, 0.68],
            "Facenet": [0.4, 0.4, 0.4, 0.4],
            "Facenet512": [0.3, 0.3, 0.3, 0.3]
        }

    # NOTE: Model Thresholds are defined in threshold_config.py
    for model in model_list:
        path = "testing_results/verification/" + model + "/"
        output_file = path + f"{mode}_Results.txt"
        with open(output_file, 'w') as f:
            f.write(f"{model}\n")
            f.write(f"{mode}Thresholds\n\n")

        for race in race_list:
            input_file = path + race + "_results.txt"
            # threshold = threshold_dict[model]
            if race == "African":
                pos = 0
            elif race == "Asian":
                pos = 1
            elif race == "Caucasian":
                pos = 2
            elif race == "Indian":
                pos = 3
                        
            threshold = threshold_dictionary[model][pos]
            print(model, race, threshold)


            write_results_to_file(race, threshold, input_file, output_file)
        print("Output file generated successfully.")
