def process_structure_file(input_file, output_file):
    # Read input file
    with open(input_file, 'r') as f_in:
        header = f_in.readline()
        lines = f_in.readlines()


    print(header)
    # Process header
    header_parts = header.strip('/t').split()
    header_parts.pop(6)
    header_parts.pop(5)
    header_parts.pop(3)

    # Process lines
    output_lines = []
    output_lines.append('\t'.join(header_parts))

    for line in lines:
        parts = line.strip('/t').split()

        parts.pop(5)
        parts.pop(3)

        output_lines.append('\t'.join(parts))


    # Write output file
    with open(output_file, 'w') as f_out:
        f_out.write('\n'.join(output_lines))

if __name__ == "__main__":
    file_list = ["Caucasian_results.txt", "Indian_results.txt"]
    model = "DeepFace"

    for file in file_list:
        input_path = "C:/Users/jay/Desktop/2_original/" + model + "/"

        output_path = "C:/Users/jay/Desktop/2_new/" + model + "/"

        process_structure_file(input_path + file, output_path + file)
        print("Output file generated successfully.")


# def process_structure_file(input_file, output_file):
#     # Read input file
#     with open(input_file, 'r') as f_in:
#         lines = f_in.readlines()
    
#     header = "Folder\tTemplate\tTest\tPredict\tTime"
#     with open(output_file, 'w') as f_out:
#         f_out.write(header + '\n')

#     counter = 0
#     combined_parts = []
#     for line in lines:
#         parts = line.strip('/t').split()
#         if counter % 2 == 0:
#             parts.pop(11)
#             parts.pop(10)
#             parts.pop(8)
#             parts.pop(7)
#             parts.pop(6)
#             parts.pop(4)
#             parts.pop(2)
#             parts.pop(0)

#             combined_parts = parts

#         if counter % 2 == 1:
#             combined_parts.append(parts[2])

#             with open(output_file, 'a') as f_out:
#                 f_out.write('\t'.join(combined_parts) + '\n')

#         counter = counter + 1

# if __name__ == "__main__":
#     file_list = ["African_results.txt", "Asian_results.txt", "Caucasian_results.txt", "Indian_results.txt"]
#     model = "DeepFace"

#     for file in file_list:
#         input_path = "C:/Users/jay/Desktop/1_original/" + model + "/"

#         output_path = "C:/Users/jay/Desktop/1_new/" + model + "/"

#         process_structure_file(input_path + file, output_path + file)
#         print("Output file generated successfully.")
