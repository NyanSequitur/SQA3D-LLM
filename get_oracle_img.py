import sys
import open3d as o3d
import argparse
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from tqdm import tqdm

def read_camera_pose(filename):
    file = open(filename, 'r')
    data = file.read()
    lines = data.split("\n")
    pose = np.array([[float(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"])
    return pose

def read_axis(filename):
    axis_align_matrix = None
    lines = open(filename).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = np.array([[float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]]).reshape(4, 4)
            break
    return axis_align_matrix

def rotate_and_trans_from_pose(pose):
    rotate = pose[:3, :3]
    trans = pose[:3, 3]
    #print("pose")
    #print(pose)
    return rotate, trans

def bbox_to_points(bbox_arr):
    x1 = bbox_arr[0] + (bbox_arr[3]/2)
    x2 = bbox_arr[0] - (bbox_arr[3]/2)
    y1 = bbox_arr[1] + (bbox_arr[4]/2)
    y2 = bbox_arr[1] - (bbox_arr[4]/2)
    z1 = bbox_arr[2] + (bbox_arr[5]/2)
    z2 = bbox_arr[2] - (bbox_arr[5]/2)
 
    points_arr = np.array([[x2, y2, z2], [x1, y2, z2], [x2, y1, z2], [x1, y1, z2], [x2, y2, z1], [x1, y2, z1], [x2, y1, z1], [x1, y1, z1]])
    return points_arr

def create_lines_from_points():
    lines = np.array([
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ])

    return lines

def bbox_info(bbox_arr, axis=None):
    points = bbox_to_points(bbox_arr)
    lines = create_lines_from_points()
    colors = np.array([[0, 200, 0] for i in range(len(lines))])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
        )
    if axis is not None:
        axis = np.array(axis, dtype=np.float)
        line_set.transform(np.linalg.inv(axis))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set.get_center()



def get_bbox_center(bbox_arr):
    return np.array([bbox_arr[0], bbox_arr[1], bbox_arr[2]])

def compute_trans_vector(trans, bbox_pos):
    return bbox_pos - trans

def compute_dist(mat1, mat2):
    return pow(mat1[0] - mat2[0], 2) + pow(mat1[1] - mat2[1], 2) + pow(mat1[2] - mat2[2], 2)

def norm_vector(vec):
    dist = pow(vec[0], 2) + pow(vec[1], 2) + pow(vec[2], 2)
    norm_vec = np.asarray([vec[0]/dist, vec[1]/dist, vec[2]/dist])
    return norm_vec

def inner_angle(vec1, vec2):
    x = np.inner(vec1,vec2)
    s = np.linalg.norm(vec1)
    t = np.linalg.norm(vec2)
    theta = np.arccos(x/(s*t))
    cos_sim = x/(s*t)
   # return theta
    return cos_sim


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_dir', help='bbox npy file')
    parser.add_argument('--frame_dir', help="camera pose")
    parser.add_argument('--save_frame')
    parser.add_argument('--cos_sim_thres')
    parser.add_argument('--output')
    parser.add_argument('--scans')
    parser.add_argument('--alignment', action='store_true')
    args = parser.parse_args()

    except_img_cnt = 0
    bboxes_list = os.listdir(args.bbox_dir)
    for bbox_file in tqdm(bboxes_list):
        ##scene0000_00_bbox.npy

        bbox_path = os.path.join(args.bbox_dir, bbox_file)
        bboxes = np.load(bbox_path)
        scene_name = str(os.path.splitext(os.path.basename(bbox_file))[0])[:12]

        ##frame_queue/scene0000_00/
        scene_path = os.path.join(args.frame_dir, scene_name)
        axis_dir = os.path.join(args.scans, scene_name)
        #axis = read_axis_matrix_from_file(os.path.join(axis_dir, scene_name + ".txt"))
        axis = read_axis(os.path.join(axis_dir, scene_name + ".txt"))
        img_dir = os.path.join(scene_path, "color")
        pose_dir = os.path.join(scene_path, "pose")
        pose_list = os.listdir(pose_dir)
        
        #for i, bbox in enumerate(bboxes):
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            #bbox_pos = get_bbox_center(bbox)
            if args.alignment:
                bbox_pos = bbox_info(bbox, axis)
            else:
                bbox_pos = bbox_info(bbox)
            obj_id = str(int(bbox[7])-1)
            obj_id = obj_id.zfill(3)
            cands = {}
            sec_cands = {}
            for pose in pose_list:
                ##frame_queue/scene0000_00/pose/0.txt
                pose_path = os.path.join(pose_dir, pose)
                cam_mat = read_camera_pose(pose_path)
                rotate, trans = rotate_and_trans_from_pose(cam_mat)
                trans_vec = compute_trans_vector(trans, bbox_pos)
                trans_dist = compute_dist(bbox_pos, trans)
                orient_vec = rotate[:3, 2]

                if inner_angle(orient_vec, trans_vec) >= float(args.cos_sim_thres):
                    cands[str(pose)]=trans_dist
                else:
                    sec_cands[str(pose)] = inner_angle(orient_vec, trans_vec)


            cands_sorted = dict(sorted(cands.items(), key=lambda x:x[1]))
            cands_pose = list(cands_sorted.keys())

            sec_cands_sorted = dict(sorted(sec_cands.items(), key=lambda x:x[1], reverse=True))
            sec_cands_pose = list(sec_cands_sorted.keys())
            
            index = 0
            while len(cands_pose) < int(args.save_frame):
                except_img_cnt += 1
                cands_pose.append(sec_cands_pose[index])
                index += 1
            
            for j in range(int(args.save_frame)):
                frame_name = os.path.splitext(os.path.basename(cands_pose[j]))[0]
                cands_img = os.path.join(img_dir, frame_name + ".jpg")
                new_img_name = str(scene_name) + "_" + str(obj_id) + "_" + str(j) + ".jpg"
                output_path = os.path.join(args.output, new_img_name)
                shutil.copy(cands_img, output_path)

    print("NO Image of this thresh")
    print(except_img_cnt)








