import json
import os

from tqdm import tqdm

bbox_json_list = []
for file in tqdm(os.listdir("./bboxes")):
    if "results" in file:
        with open(os.path.join("./bboxes", file), "r") as f:
            bbox_json = json.load(f)
        bbox_json_list.append(bbox_json)

full_bbox_json = {}
count = 0
for bbox_json in tqdm(bbox_json_list):
    for file_name, file_name_json in bbox_json.items():
        if file_name not in full_bbox_json.keys():
            full_bbox_json[file_name] = {}
        for _, ep_json in file_name_json.items():
            ep_id = ep_json["episode_id"]
            if ep_id in full_bbox_json[file_name].keys():
                print(ep_id, ep_json)
                print(full_bbox_json[file_name][ep_id])
                raise ValueError("Duplicate episode id")
            full_bbox_json[file_name][ep_id] = ep_json
            count += 1

print(count)
with open(os.path.join("./bboxes", "full_bboxes.json"), "w") as f:
    json.dump(full_bbox_json, f)
