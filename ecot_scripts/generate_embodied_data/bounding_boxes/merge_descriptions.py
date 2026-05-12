import json
import os

full_json = {}
count = 0

path_to_desc = "./descriptions"
path_to_full_desc = os.path.join(path_to_desc, "full_descriptions.json")

for json_f in os.listdir(path_to_desc):
    if "full" in json_f:
        continue
    with open(os.path.join(path_to_desc, json_f), "r") as f:
        desc_json = json.load(f)

    for file_path, file_json in desc_json.items():
        if file_path not in full_json.keys():
            full_json[file_path] = {}
        for ep_id, desc_dict in file_json.items():
            assert ep_id not in full_json[file_path].keys()
            full_json[file_path][ep_id] = desc_dict
            count += 1

print("Inserted", count, "trajectory descriptions into combined json")
print("Saving to:", path_to_full_desc)

with open(path_to_full_desc, "w") as f:
    json.dump(full_json, path_to_full_desc)
