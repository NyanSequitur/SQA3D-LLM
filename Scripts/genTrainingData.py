import os
import torch
import pandas as pd
import json


def get_camera_pose_dic(pose_files):
    camera_pose_dic = {
        int(os.path.basename(pose_file).strip(".txt")): read_camera_pose(pose_file)
        for pose_file in pose_files
    }

    return camera_pose_dic


def get_bbox_image(
    bbox_centers_list,
    pose_ids_list,
    camera_pose_all_list,
    num_frame=1,
    cossim_thresh=0.8,
):
    N = len(bbox_centers_list)
    num_box_pose_list = []

    camera_rotate_parts = []
    camera_trans_parts = []
    bbox_centers_parts = []
    for i in range(N):
        num_box = bbox_centers_list[i].shape[0]
        num_pose = len(pose_ids_list[i])
        num_box_pose_list.append((num_box, num_pose))
        camera_pose_part = (
            camera_pose_all_list[i]
            .unsqueeze(0)
            .expand(num_box, num_pose, 4, 4)
            .reshape(-1, 4, 4)
        )
        camera_rotate_part = camera_pose_part[:, :3, :3]
        camera_trans_part = camera_pose_part[:, :3, 3]
        bbox_centers_part = (
            bbox_centers_list[i]
            .unsqueeze(1)
            .expand(num_box, num_pose, 3)
            .reshape(-1, 3)
        )

        camera_rotate_parts.append(camera_rotate_part)
        camera_trans_parts.append(camera_trans_part)
        bbox_centers_parts.append(bbox_centers_part)

    camera_rotate_all = torch.cat(camera_rotate_parts)

    camera_trans_all = torch.cat(camera_trans_parts)

    bbox_centers = torch.cat(bbox_centers_parts)

    trans_dist = torch.norm(camera_trans_all - bbox_centers, dim=-1)

    trans_vec_all = bbox_centers - camera_trans_all

    orient_vec_all = camera_rotate_all[:, :3, 2]

    cossim = (
        (orient_vec_all * trans_vec_all).sum(dim=-1)
        / torch.norm(orient_vec_all, dim=-1)
        / torch.norm(trans_vec_all, dim=-1)
    )

    trans_closeness = 1 + (1 - trans_dist / trans_dist.max())

    trans_closeness[cossim <= cossim_thresh] = 0
    cossim[cossim > cossim_thresh] = 0

    score = cossim + trans_closeness

    bbox_pose_ids_list = []
    curr = 0
    for i, (num_box, num_pose) in enumerate(num_box_pose_list):
        num_box_pose = num_box * num_pose
        score_part = score[curr : (curr + num_box_pose)].reshape(num_box, num_pose)
        curr += num_box_pose
        print(score_part.shape)
        score_topk_value, score_topk_index = torch.topk(
            score_part, min(num_frame, score_part.shape[1]), dim=1
        )
        print("score_topk_index", score_topk_index.shape)
        print("pose_ids_list[i]", pose_ids_list[i].shape)

        bbox_pose_ids_list.append(pose_ids_list[i][score_topk_index])

    return bbox_pose_ids_list


if __name__ == "__main__":
    # sys.path.append(".")
    from get_oracle_img import *

    data_split = "train"  # val
    scanrefer_dir = ""
    scanrefer_file = os.path.join(
        scanrefer_dir, "ScanRefer_filtered_" + data_split + ".json"
    )
    ref_df = pd.read_json(scanrefer_file)
    print(ref_df)
    scene_file_list_file_path = os.path.join(
        scanrefer_dir, "ScanRefer_filtered_" + data_split + ".txt"
    )

    scene_file_list_file = open(scene_file_list_file_path, "r")
    scene_file_list = scene_file_list_file.read()
    scene_ids = scene_file_list.split("\n")
    scene_file_list_file.close()

    data_dir = "/home/dbi-data5/miyanishi/Project/ScanRefer/data/scannet/"
    
    scannet_data_dir = os.path.join(data_dir, "scannet_data")
    frame_dir = "/home/dbi-data5/miyanishi/Project/ScanRefer/data/frames_square"

    camera_pose_all_dic = {}
    bbox_all_dic = {}

    for scene_id in scene_ids:
        print(scene_id + "/" + scene_ids[-1])
        bbox_file = os.path.join(scannet_data_dir, scene_id + "_bbox.npy")
        bboxes = np.load(bbox_file)

        scene_frame_dir = os.path.join(frame_dir, scene_id)

        pose_dir = os.path.join(scene_frame_dir, "pose")
        pose_files = [
            os.path.join(pose_dir, pose_fname) for pose_fname in os.listdir(pose_dir)
        ]

        bbox_all_dic[scene_id] = bboxes
        camera_pose_all_dic[scene_id] = get_camera_pose_dic(pose_files)

    num_frame = 10
    cossim_thresh = 0.9

    bbox_centers_list = []
    camera_pose_ids_list = []
    camera_poses_list = []

    for scene_id in scene_ids:
        bbox_centers = []
        bboxes_tensor = torch.from_numpy(bbox_all_dic[scene_id])

        tmp_df = ref_df.groupby(["scene_id"]).count().reset_index()
        N = {
            scene_id: num_caption
            for (scene_id, num_caption) in zip(
                list(tmp_df.scene_id), list(tmp_df.description)
            )
        }[scene_id]

        print(bboxes_tensor[:, 7])

        for i in range(N):
            record = ref_df[ref_df.scene_id == scene_id].iloc[i]
            bbox = bboxes_tensor[bboxes_tensor[:, 7] == record.object_id][0]
            bbox_centers.append(bbox[:3])

        bbox_centers = torch.stack(bbox_centers)
        bbox_centers_list.append(bbox_centers)
        camera_pose_dic = camera_pose_all_dic[scene_id]
        camera_pose_ids = np.array(list(camera_pose_dic.keys()))
        camera_poses = torch.from_numpy(np.stack(list(camera_pose_dic.values())))

        camera_pose_ids_list.append(camera_pose_ids)
        camera_poses_list.append(camera_poses)

    image_pose_ids_list = get_bbox_image(
        bbox_centers_list,
        camera_pose_ids_list,
        camera_poses_list,
        num_frame=num_frame,
        cossim_thresh=cossim_thresh,
    )

    dataSetJson = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Train

    if data_split == "train":
        dataSetJson = []
        for i in range(len(scene_ids)):
            scene_id = scene_ids[i]
            image_pose_ids = image_pose_ids_list[i]
            for j in range(len(ref_df[ref_df.scene_id == scene_id])):
                record = ref_df[ref_df.scene_id == scene_id].iloc[j]
                for k in image_pose_ids[j]:
                    dataSetJson.append(
                        {
                            "caption": record.description,
                            "image": f"{scene_id}/exported/color/{k}.jpg",
                            "image_id": f"{''.join(char for char in scene_id if char.isdigit())}{k}",
                        }
                    )

        # Sort dataSetJson by scene ID and image number
        dataSetJson.sort(
            key=lambda x: (
                x["image"].split("/")[0],
                int(x["image"].split("/")[3].split(".")[0]),
            )
        )

        # Write the modified dataSetJson to a file
        jsonObj = json.dumps(dataSetJson)
        with open(f"{data_split}.json", "w") as outfile:
            outfile.write(jsonObj)

    # GT
    elif data_split == "val":
        dataSetJson = {"info": {}, "licenses": [], "images": [], "annotations": []}
        for i in range(len(scene_ids)):
            scene_id = scene_ids[i]
            image_pose_ids = image_pose_ids_list[i]
            for j in range(len(ref_df[ref_df.scene_id == scene_id])):
                record = ref_df[ref_df.scene_id == scene_id].iloc[j]
                for k in image_pose_ids[j]:
                    image_filename = f"{scene_id}/exported/color/{k}.jpg"
                    image_id = int(
                        "".join(char for char in image_filename if char.isdigit())
                    )

                    dataSetJson["images"].append(
                        {
                            "id": image_id,
                            "file_name": image_filename,
                            "caption": record.description,
                        }
                    )

                    dataSetJson["annotations"].append(
                        {
                            "id": len(dataSetJson["annotations"]) + 1,
                            "image_id": image_id,
                            "caption": record.description,
                        }
                    )

        # Sort dataSetJson by image ID
        dataSetJson["images"].sort(key=lambda x: x["id"])

        # Write the modified dataSetJson to a file
        with open(f"{data_split}_gt.json", "w") as json_file:
            json.dump(dataSetJson, json_file)

        dataSetJson = []
        # Iterate over the scene_ids
        for i in range(len(scene_ids)):
            scene_id = scene_ids[i]
            image_pose_ids = image_pose_ids_list[i]

            # Iterate over the records for the current scene_id
            for j in range(len(ref_df[ref_df.scene_id == scene_id])):
                record = ref_df[ref_df.scene_id == scene_id].iloc[j]
                image_pose_id = image_pose_ids[j]
                for image in image_pose_id:
                    # Generate a unique image_id
                    # image_id = str(uuid.uuid4())

                    # Generate the image path
                    image_path = f"{scene_id}/exported/color/{image}.jpg"

                    # Check if the image ID already exists in dataSetJson
                    existing_image = next(
                        (item for item in dataSetJson if item["image"] == image_path),
                        None,
                    )

                    if existing_image:
                        # Append caption to the existing caption list
                        existing_image["caption"].append(record.description)
                    else:
                        # Add new image and caption to dataSetJson
                        dataSetJson.append(
                            {
                                # "image_id": image_id,
                                "image": image_path,
                                "caption": [record.description],
                            }
                        )

        # Sort dataSetJson based on image_id
        # dataSetJson = sorted(dataSetJson, key=lambda x: x["image_id"])

        # Write the modified dataSetJson to a file
        jsonObj = json.dumps(dataSetJson)
        with open(f"{data_split}.json", "w") as outfile:
            outfile.write(jsonObj)