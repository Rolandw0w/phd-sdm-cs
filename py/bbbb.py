import os


for dir_path, dir_names, file_names in os.walk("/home/rolandw0w/Development/PhD/output/synth/jaeckel"):
    read_files = [file_name for file_name in file_names if file_name.startswith("read")]
    if not read_files:
        continue

    for read_file in read_files:
        path = f"{dir_path}/{read_file}"
        print(path)

        with open(path, "r") as f_read:
            content = f_read.read()
            l_c = len(content)

        no_comma_content = content.replace(",", "")
        lnc = len(no_comma_content)
        with open(path, "w") as f_write:
            f_write.write(no_comma_content)
            print()
