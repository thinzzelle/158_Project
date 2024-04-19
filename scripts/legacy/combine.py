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


def process_structure_file(file1, file2, output_file):
    # Read input file
    with open(file1, 'r') as f_in:
        header = f_in.readline()
        lines1 = f_in.readlines()

    with open(file2, 'r') as f_in:
        header = f_in.readline()
        lines2 = f_in.readlines()
 
    print(len(lines1))
   
    with open(output_file, 'w') as f_out:
        f_out.write(header)
        for line in lines1:
            f_out.write(line)
        # f_out.write("\n")
        for line in lines2:
            f_out.write(line)


if __name__ == "__main__":

    race_list = ["African", "Asian", "Caucasian", "Indian"]
    model = "DeepFace"
    output_path = "C:/Users/jay/Desktop/combined/"+ model + "/"

    for race in race_list:
        file1 = "C:/Users/jay/Desktop/1_new/" + model + "/" + race + "_results.txt"
        file2 = "C:/Users/jay/Desktop/2_new/" + model + "/" + race + "_results.txt"
        output_file = output_path + race + "_results.txt" 

        file = race + "_results.txt"
        process_structure_file(file1, file2, output_file)
    print("Output file generated successfully.")

