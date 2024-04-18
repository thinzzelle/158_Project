import matplotlib.pyplot as plt

# Step 1: Read data from the text file
# Assuming the data is stored in a file named 'face_data.txt'

positive_pairs = []
negative_pairs = []

with open('C:/Users/jay/Desktop/African_results.txt', 'r') as file:
    file.readline()  # Skip the header
    count = 0
    for line in file:
        parts = line.split()
        distance = parts[7]
        distance = float(distance [:7])
        
        # print(parts[0], parts[2], distance)
        if parts[0] == parts[2]:
            positive_pairs.append(distance)
        else:
            negative_pairs.append(distance)

        count += 1
        if count > 10000:
            break


fpr_list = []
fnr_list = []
thresholds = [(i / 1000) for i in range(1001)]


for threshold in thresholds:
    false_positive = sum(1 for distance in positive_pairs if distance > threshold)
    true_positive = len(positive_pairs) - false_positive
    false_negative = sum(1 for distance in negative_pairs if distance < threshold)
    true_negative = len(negative_pairs) - false_negative


    if (false_positive + true_negative) == 0:
        fpr = 0
    else:
        fpr = false_positive / (false_positive + true_negative)

    if (true_positive + false_negative) == 0:
        fnr = 0
    else:
        fnr = false_negative / (false_negative + true_positive)
    
    fpr_list.append(fpr)
    fnr_list.append(fnr)


# Step 3: Plot FPR and FNR
plt.plot(thresholds, fpr_list, label='False Positive Rate (FPR)')
plt.plot(thresholds, fnr_list, label='False Negative Rate (FNR)')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('Equal Error Rate (EER)')
plt.legend()
plt.xticks([i / 10 for i in range(11)])  # Set x-axis ticks at increments of 0.1
plt.grid(True)
plt.show()
