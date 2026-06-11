input_file_path  = "./cls_test.txt"
output_file_path = "./testlist01.txt"


with open(input_file_path, 'r') as file:
    content = file.read()


modified_content = content.replace("Sad", "Sadness")


with open(output_file_path, 'w') as file:
    file.write(modified_content)

print(f"File has been modified. Modified content saved to {output_file_path}")
