import os
import argparse

time_dict = {}
count = 0
time_index = 3  # index of timestamp in your input lines


def preprocess(data_part, path, newpath):
    global count
    data_path = os.path.join(path, f"{data_part}.txt")
    write_path = os.path.join(newpath, f"{data_part}.txt")

    with open(data_path) as fp, open(write_path, "w") as fw:
        for line in fp:
            count += 1
            info = line.strip().split("\t")

            time_val = info[time_index]
            if time_val not in time_dict:
                time_dict[time_val] = len(time_dict)
            time_id = time_dict[time_val]

            fw.write(f"{int(info[0])} {int(info[1])} {int(info[2])} {time_id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert time column to indexed time IDs.")
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Name of the dataset folder')
    parser.add_argument('--datapath', '-p', type=str, required=True, help='Path to the dataset directory')

    args = parser.parse_args()
    dataset = args.dataset
    base_path = args.datapath
    new_path = os.path.join(args.datapath, f"{dataset}")

    os.makedirs(new_path, exist_ok=True)

    # Process stat.txt
    with open(os.path.join(base_path, "stat.txt"), "r") as fr_stat, open(os.path.join(new_path, "stat.txt"), "w") as fw_stat:
        entity_total, relation_total, _ = fr_stat.readline().split()

        preprocess("train2id", base_path, new_path)
        preprocess("valid2id", base_path, new_path)
        preprocess("test2id", base_path, new_path)

        print(f"✅ Total processed lines: {count}")
        fw_stat.write(f"{entity_total} {relation_total} {len(time_dict)}")

