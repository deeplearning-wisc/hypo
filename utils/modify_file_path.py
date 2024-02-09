ORIGINAL_ROOT_DIR = '/nobackup/hybai/datasets/Generalization/PACS'
NEW_ROOT_DIR = 'datasets'

domains = ['art_painting', 'cartoon', 'photo', 'sketch']
for domain in domains:
    input_file = f"{NEW_ROOT_DIR}/pacs_data/{domain}.txt"
    output_file = f"{NEW_ROOT_DIR}/pacs_data/{domain}.txt"
    with open(input_file, "r") as f:
        lines = f.readlines()
    updated_lines = [line.replace(ORIGINAL_ROOT_DIR, NEW_ROOT_DIR) for line in lines]
    with open(output_file, "w") as f:
        f.writelines(updated_lines)
