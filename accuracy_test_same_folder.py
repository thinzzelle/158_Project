model_list = ["DeepFace", "ArcFace", "Facenet", "Facenet512"]
race_list = ["African", "Asian", "Caucasian", "Indian"]

with open ('temporary_results.txt', 'w') as f_out:
    pass

for model in model_list:
    for race in race_list:

        with open(f'testing_results/same_folder_test/{model}/{race}_results.txt', 'r') as    f_in:
            header = f_in.readline()
            lines = f_in.readlines()

        tp = 0

        # count = 0
        for line in lines:
            parts = line.strip('/t').split()
            # print('parts 3', parts[3] )
            if int(parts[3]) == 1:
                tp += 1

            # count += 1
            # if count == 10:
            #     break
            
        with open('temporary_results.txt', 'a') as f_out:
            f_out.write(f'{model} - {race}')
            f_out.write(f'\n\tTrue Positive:\t{tp}\n')
            f_out.write(f'\tTotal Test Count\t{len(lines)}\n')
            f_out.write(f'\tRecall:\t{tp/len(lines)}\n\n')
            
        
