# calculates the race results for a set of files. The code reads the input file, processes the header and lines, and writes the output file. The code is executed for each file in the file_list list. The input and output paths are hard-coded in the script.

def _calculate_scores(tp, fp, tn, fn, test_count):

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


def process_structure_file(race, input_file, output_file):
    # Read input file
    with open(input_file, 'r') as f_in:
        header = f_in.readline()
        lines = f_in.readlines()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    positive_test = 0
    negative_test = 0

    # Process lines
    for line in lines:
        parts = line.strip('/t').split()

        if parts[3] == str(1):
            tp = tp + 1
        if parts[3] == str(0):
            fn = fn + 1
        
        positive_test = positive_test + 1

    
    # Calculate accuracy
    f1_score, accuracy, recall, precision, specificity = _calculate_scores(tp, fp, tn, fn, positive_test)

    

    # Write output file
    with open(output_file, 'a') as f_out:
        f_out.write(race)
        f_out.write(f"\n\tTrue Positive:\t{tp}\n\tTrue Negative:\t{tn}\n\tFale  Positive:\t{fp}\n\tFalse Negative:\t{fn}\n")
        f_out.write(f"\tPositive Test Count:\t{positive_test}\n")
        f_out.write(f"\tNegative Test Count:\t{negative_test}\n\n")
        f_out.write("\tTotal Test Time:\n")
        f_out.write("\tStart Time:\n")
        f_out.write("\tEnd Time:\n")
        f_out.write("\tAvg Test Time:\n\n")

        f_out.write(f"\tf1 score:\t{f1_score}\n\taccuracy:\t{accuracy}\n\trecall:\t\t{recall}\n\tprecision:\t{precision}\n\tspecificity:\t{specificity}\n\n")

if __name__ == "__main__":

    race_list = ["African", "Asian", "Caucasian", "Indian"]
    model = "DeepFace"
    input_path = "C:/Users/jay/Desktop/combined/" + model + "/"
    

    output_path = "C:/Users/jay/Desktop/combined/" + model + "/"
    output_file = output_path + "Race_results.txt" 

    with open(output_file, 'w') as f_out:
        pass

    for race in race_list:
        file = race + "_results.txt"
        process_structure_file(race, input_path + file, output_file)
    print("Output file generated successfully.")

