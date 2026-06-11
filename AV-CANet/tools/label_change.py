data_file = './cls_train.txt'
output_file = './cls_train_modified.txt'


with open(data_file, 'r') as file:
    lines = file.readlines()


modified_lines = []
for line in lines:
    line = line.strip()
    if line:
        sample_path, label = line.split(' ')
        label = str(int(label) - 1)
        modified_line = f"{sample_path} {label}\n"
        modified_lines.append(modified_line)


with open(output_file, 'w') as file:
    file.writelines(modified_lines)

print(f"Modified data saved to {output_file}")
