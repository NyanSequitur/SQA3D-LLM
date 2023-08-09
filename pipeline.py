import logging, sys

# log to log.log
logging.basicConfig(
    filename="log.log",
    filemode="w",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.DEBUG,
)

logging.debug("Starting pipeline.py")

from SegmentAnything3D.util import (
    pairwise_indices,
    num_to_natural,
    remove_small_group,
    Voxelize,
)

logging.debug("Imported SegmentAnything3D.util")

from SegmentAnything3D.sam3d import (
    cal_2_scenes,
    SamAutomaticMaskGenerator,
    build_sam,
    cal_group,
    get_sam,
)

logging.debug("Imported SegmentAnything3D.sam3d")

import subprocess

logging.debug("Imported subprocess")

import shutil

logging.debug("Imported shutil")

import numpy as np

logging.debug("Imported numpy")

from PIL import Image

logging.debug("Imported Image")

from os.path import join, isfile, basename

logging.debug("Imported join, isfile, basename")

import json

logging.debug("Imported json")

import os

logging.debug("Imported os")

from torchvision.transforms.functional import resize

logging.debug("Imported resize")

from tqdm import tqdm

logging.debug("Imported tqdm")

from concurrent.futures import ThreadPoolExecutor

logging.debug("Imported ThreadPoolExecutor")

import cv2

logging.debug("Imported cv2")

import SegmentAnything3D

logging.debug("Imported SegmentAnything3D")

import pointops

logging.debug("Imported pointops")

from get_oracle_img import read_camera_pose

logging.debug("Imported read_camera_pose")

from omegaconf import OmegaConf

logging.debug("Imported OmegaConf")

from lavis.common.registry import registry

logging.debug("Imported registry")

from lavis.models import load_preprocess

logging.debug("Imported load_preprocess")

import torch.multiprocessing as mp

logging.debug("Imported mp")

import torch

logging.debug("Imported torch")

import torch.multiprocessing as mp

logging.debug("Imported mp")

from ChatCaptioner.chatcaptioner.blip2 import Blip2

from ChatCaptioner.chatcaptioner.chat import (
    set_openai_key,
    caption_image,
)

from dotenv import load_dotenv
# load environment variables
load_dotenv()
set_openai_key(os.getenv("OPENAI_API_KEY"))


# MSIC. UTILS


def executeConsoleCommand(command, args):
    command = [command] + args
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True
    ) as p:
        for line in p.stdout:
            print(line, end="")


def runPythonScript(scriptPath, args):
    command = ["python", scriptPath] + args
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True
    ) as p:
        for line in p.stdout:
            print(line, end="")


def getTrainAndValLists(trainPath, valPath):
    with open(trainPath) as trainFile:
        trainScenes = trainFile.read().splitlines()
    with open(valPath) as valFile:
        valScenes = valFile.read().splitlines()
    return trainScenes, valScenes


def getSplitOfScene(sceneID, trainPath, valPath):
    trainScenes, valScenes = getTrainAndValLists(trainPath, valPath)
    if sceneID in trainScenes:
        return "train"
    elif sceneID in valScenes:
        return "val"
    else:
        return "test"


# FILE GENERATION UTILS
def createSegmentatorExecutable():
    if not isfile(join("Segmentator", "segmentator")):
        command = "make"
        args = ["--directory", "Segmentator"]
        executeConsoleCommand(command, args)


def generateSegsFile(scansPath, sceneID):
    segmentatorPath = join("Segmentator", "segmentator")
    command = f"./{segmentatorPath}"
    args = [join(scansPath, sceneID, f"{sceneID}_vh_clean_2.ply")]
    executeConsoleCommand(command, args)


# PREPROCESSING
def perform2DDataPreprocessing(scansPath, rgbPath, width, height):
    segmentAnythingPath = SegmentAnything3D.__path__[0]
    scriptPath = join(
        segmentAnythingPath,
        "scannet-preprocess/prepare_2d_data/prepare_2d_data.py",
    )
    args = [
        "--scannet_path",
        scansPath,
        "--output_path",
        rgbPath,
        "--output_image_width",
        str(width),
        "--output_image_height",
        str(height),
    ]
    runPythonScript(scriptPath, args)
    for scene in os.listdir(rgbPath):
        shutil.copytree(
            f"{scansPath}/{scene}/exported/intrinsic",
            f"{rgbPath}/{scene}/intrinsics",
            dirs_exist_ok=True,
        )


def performScannetPreprocessing(scanNetPath, dataPath):
    segmentAnythingPath = SegmentAnything3D.__path__[0]
    scriptPath = join(segmentAnythingPath, "scannet-preprocess/preprocess_scannet.py")
    args = ["--dataset_root", scanNetPath, "--output_root", dataPath]
    runPythonScript(scriptPath, args)


def get_pcd(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path):
    intrinsic_path = join(rgb_path, scene_name, "intrinsics", "intrinsic_depth.txt")
    depth_intrinsic = np.loadtxt(intrinsic_path)
    pose = join(rgb_path, scene_name, "pose", color_name[0:-4] + ".txt")
    depth = join(rgb_path, scene_name, "depth", color_name[0:-4] + ".png")
    color = join(rgb_path, scene_name, "color", color_name)
    depth_img = cv2.imread(depth, -1)  # read 16-bit grayscale image
    mask = np.nonzero(depth_img)
    color_image = cv2.imread(color)
    color_image = cv2.resize(color_image, (640, 480))  # Resize color image
    save_2dmask_path = join(save_2dmask_path, scene_name)
    if mask_generator is not None:
        group_ids = get_sam(color_image, mask_generator)
        if not os.path.exists(save_2dmask_path):
            os.makedirs(save_2dmask_path)
        img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode="I;16")
        img.save(join(save_2dmask_path, color_name[0:-4] + ".png"))
    else:
        group_path = join(save_2dmask_path, color_name[0:-4] + ".png")
        img = Image.open(group_path)
        group_ids = np.array(img, dtype=np.int16)
    colors = color_image[mask][:, ::-1]  # Reorder color channels
    group_ids = num_to_natural(group_ids[mask])
    pose = np.loadtxt(pose)
    depth_shift = 1000.0
    width, height = depth_img.shape[1], depth_img.shape[0]
    x, y = np.meshgrid(
        np.arange(width),
        np.arange(height),
    )
    uv_depth = np.column_stack(
        (x.flatten(), y.flatten(), depth_img.flatten() / depth_shift)
    )
    uv_depth = uv_depth[uv_depth[:, 2] != 0]
    fx, fy, cx, cy, bx, by = (
        depth_intrinsic[0, 0],
        depth_intrinsic[1, 1],
        depth_intrinsic[0, 2],
        depth_intrinsic[1, 2],
        depth_intrinsic[0, 3],
        depth_intrinsic[1, 3],
    )
    n = uv_depth.shape[0]
    points = np.empty((n, 4))
    points[:, 0] = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
    points[:, 1] = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
    points[:, 2] = uv_depth[:, 2]
    points[:, 3] = 1
    points_world = np.dot(points, pose.T)
    save_dict = dict(coord=points_world[:, :3], color=colors, group=group_ids)
    return save_dict


def process_color(
    scene_name, color_names, rgb_path, mask_generator, save_2dmask_path, voxelize
):
    pcd_list = []
    for color_name in tqdm(color_names, position=0):
        pcd_dict = get_pcd(
            scene_name, color_name, rgb_path, mask_generator, save_2dmask_path
        )
        if len(pcd_dict["coord"]) > 0:
            pcd_list.append(voxelize(pcd_dict))
    # release shared CUDA tensors
    return pcd_list


def seg_pcd(
    scene_name,
    rgb_path,
    data_path,
    save_path,
    mask_generator,
    voxel_size,
    voxelize,
    th,
    train_scenes,
    val_scenes,
    save_2dmask_path,
):
    print(scene_name)
    if os.path.exists(join(save_path, scene_name + ".pth")):
        return
    color_names = sorted(
        os.listdir(join(rgb_path, scene_name, "color")),
        key=lambda a: int(os.path.basename(a).split(".")[0]),
    )
    num_processes = 4
    chunk_size = (len(color_names) + num_processes - 1) // num_processes
    # Create a pool of processes
    pool = mp.Pool(processes=num_processes)
    # Divide color_names into chunks for each process
    chunks = [
        color_names[i : i + chunk_size] for i in range(0, len(color_names), chunk_size)
    ]
    # Map the color_name chunks to the process_color function using the process pool
    results = []
    for chunk in chunks:
        result = pool.apply_async(
            process_color,
            args=(
                scene_name,
                chunk,
                rgb_path,
                mask_generator,
                save_2dmask_path,
                voxelize,
            ),
        )
        results.append(result)
    # Close the pool to indicate that no more tasks will be submitted
    pool.close()
    # Wait for all processes to complete and get the results
    pcd_lists = [result.get() for result in results]
    # Flatten the pcd_lists into a single list
    pcd_list = [pcd for sublist in pcd_lists for pcd in sublist]
    while len(pcd_list) > 1:
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            pcd_frame = cal_2_scenes(
                pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize
            )
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))
    # MODIFIED
    scene_path = join(data_path, "test", scene_name + ".pth")
    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    data_dict = torch.load(scene_path)
    scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    group[mask_dis] = -1
    group = group.astype(np.int16)
    torch.save(num_to_natural(group), join(save_path, scene_name + ".pth"))
    torch.cuda.ipc_collect()


def seg_pcd_no_multiprocessing(
    scene_name,
    rgb_path,
    data_path,
    save_path,
    mask_generator,
    voxel_size,
    voxelize,
    th,
    train_scenes,
    val_scenes,
    save_2dmask_path,
):
    print(scene_name)
    if os.path.exists(join(save_path, scene_name + ".pth")):
        return
    color_names = sorted(
        os.listdir(join(rgb_path, scene_name, "color")),
        key=lambda a: int(os.path.basename(a).split(".")[0]),
    )
    pcd_list = process_color(
        scene_name,
        color_names,
        rgb_path,
        mask_generator,
        save_2dmask_path,
        voxelize,
    )
    while len(pcd_list) > 1:
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            pcd_frame = cal_2_scenes(
                pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize
            )
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))
    # MODIFIED
    scene_path = join(data_path, "test", scene_name + ".pth")
    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    data_dict = torch.load(scene_path)
    scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    group[mask_dis] = -1
    group = group.astype(np.int16)
    torch.save(num_to_natural(group), join(save_path, scene_name + ".pth"))
    torch.cuda.ipc_collect()


def perform3DSegmentation(
    trainPath,
    valPath,
    checkpointPath,
    savePath,
    rgbPath,
    dataPath,
    save2dMaskPath,
    voxelSize,
    threshold,
):
    # set mp method to spawn
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()
    sceneNames = sorted(os.listdir(rgbPath))
    # If no GPUs are available, fallback to CPU
    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        device = torch.device("cpu")
    else:
        print(
            f"Detected {num_gpus} GPUs. Running on {min(num_gpus, len(sceneNames))} GPUs."
        )
    trainScenes, valScenes = getTrainAndValLists(trainPath, valPath)
    # Determine the number of scenes and the number of scenes per GPU
    num_scenes = len(sceneNames)
    scenes_per_gpu = num_scenes // num_gpus
    remainder_scenes = num_scenes % num_gpus
    processes = []
    voxelize = Voxelize(
        voxel_size=voxelSize, mode="train", keys=("coord", "color", "group")
    )
    start_idx = 0
    for i in range(num_gpus):
        num_assigned_scenes = scenes_per_gpu + int(i < remainder_scenes)
        end_idx = start_idx + num_assigned_scenes
        scene_names_slice = sceneNames[start_idx:end_idx]
        start_idx = end_idx
        device_i = torch.device(f"cuda:{i}")
        if scene_names_slice:
            print(f"Process {i}: Handling {len(scene_names_slice)} scenes")
            process = mp.Process(
                target=process_instance,
                args=(
                    scene_names_slice,
                    rgbPath,
                    dataPath,
                    savePath,
                    checkpointPath,
                    voxelSize,
                    threshold,
                    trainScenes,
                    valScenes,
                    save2dMaskPath,
                    device_i,
                    voxelize,
                ),
            )
            processes.append(process)
            process.start()
    for process in processes:
        process.join()


def process_instance(
    sceneNames,
    rgbPath,
    dataPath,
    savePath,
    checkpointPath,
    voxelSize,
    threshold,
    trainScenes,
    valScenes,
    save2dMaskPath,
    device_i,
    voxelize,
):
    # Set the current device for the process
    torch.cuda.set_device(device_i)
    print(f"Process {device_i}: Loading SAM")
    maskGenerator = SamAutomaticMaskGenerator(
        build_sam(checkpoint=checkpointPath).to(device=device_i)
    )
    print(f"Process {device_i}: Starting segmentation")
    for sceneName in sceneNames:
        with torch.cuda.device(device_i):
            seg_pcd(
                sceneName,
                rgbPath,
                dataPath,
                savePath,
                maskGenerator,
                voxelSize,
                voxelize,
                threshold,
                trainScenes,
                valScenes,
                save2dMaskPath,
            )


def perform3DSegmentationNoMultiprocessing(
    trainPath,
    valPath,
    checkpointPath,
    savePath,
    rgbPath,
    dataPath,
    save2dMaskPath,
    voxelSize,
    threshold,
):
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    sceneNames = sorted(os.listdir(rgbPath))
    print("creating voxelize")
    voxelize = Voxelize(
        voxel_size=voxelSize, mode="train", keys=("coord", "color", "group")
    )
    print("creating mask generator")
    device = torch.device("cuda:0")
    print("building sam")
    sam = build_sam(checkpoint=checkpointPath)
    print("moving sam to device")
    sam.to(device=device)
    print("creating mask generator")
    maskGenerator = SamAutomaticMaskGenerator(sam)
    print("getting train and val lists")
    trainScenes, valScenes = getTrainAndValLists(trainPath, valPath)
    print("starting segmentation")
    for sceneName in tqdm(sceneNames, position=1):
        seg_pcd_no_multiprocessing(
            sceneName,
            rgbPath,
            dataPath,
            savePath,
            maskGenerator,
            voxelSize,
            voxelize,
            threshold,
            trainScenes,
            valScenes,
            save2dMaskPath,
        )


# BBOX GENERATION


def generateBoundingBoxes(data, newGroup):
    numPoints = data.shape[0]
    uniqueGroups = np.unique(newGroup)
    bboxes = []
    for group in uniqueGroups:
        groupIndices = np.where(newGroup == group)[0]
        groupPoints = data[groupIndices]
        minCoords = np.min(groupPoints, axis=0)
        maxCoords = np.max(groupPoints, axis=0)
        cx = (minCoords[0] + maxCoords[0]) / 2.0
        cy = (minCoords[1] + maxCoords[1]) / 2.0
        cz = (minCoords[2] + maxCoords[2]) / 2.0
        dx = maxCoords[0] - minCoords[0]
        dy = maxCoords[1] - minCoords[1]
        dz = maxCoords[2] - minCoords[2]
        semantic_label = group
        bbox = [cx, cy, cz, dx, dy, dz, semantic_label]
        bboxes.append(bbox)
    return np.array(bboxes)


def createNpyFiles(orgPath, newPath, dataPath, sceneID, scansPath):
    newPointCloud = torch.load(newPath)
    newPointCloud = num_to_natural(remove_small_group(newPointCloud, 20))
    with open(orgPath) as f:
        segments = json.load(f)
        orgPointCloud = np.array(segments["segIndices"])
    match_inds = [(i, i) for i in range(len(newPointCloud))]
    newGroup = cal_group(
        dict(group=newPointCloud), dict(group=orgPointCloud), match_inds
    )
    print(newGroup.shape)
    data = torch.load(dataPath)["coord"]
    bounding_boxes = generateBoundingBoxes(data, newGroup)
    np.save(join(scansPath, sceneID, f"{sceneID}_bbox.npy"), bounding_boxes)


def generateBboxEnsembles(
    scansPath, sceneIDs, trainPath, valPath, savePath, preProcessOutputPath
):
    for sceneID in sceneIDs:
        org = join(scansPath, sceneID, f"{sceneID}_vh_clean_2.0.010000.segs.json")
        if not isfile(org):
            generateSegsFile(scansPath, sceneID)
        split = getSplitOfScene(sceneID, trainPath, valPath)
        new = join(savePath, f"{sceneID}.pth")
        data = join(preProcessOutputPath, split, f"{sceneID}.pth")
        createNpyFiles(org, new, data, sceneID, scansPath)


# FRAMES LIST GENERATION


def getCameraPoseDic(poseFiles):
    cameraPoseDic = {
        int(basename(poseFile).strip(".txt")): read_camera_pose(poseFile)
        for poseFile in poseFiles
    }

    return cameraPoseDic


def getBboxImage(
    bboxCentersList,
    poseIDsList,
    cameraPoseAllList,
    numFrame=1,
    cossimThresh=0.8,
):
    N = len(bboxCentersList)
    numBoxPoseList = []

    cameraRotateParts = []
    cameraTransParts = []
    bboxCentersParts = []
    for i in range(N):
        numBox = bboxCentersList[i].shape[0]
        numPose = len(poseIDsList[i])
        numBoxPoseList.append((numBox, numPose))
        cameraPosePart = (
            cameraPoseAllList[i]
            .unsqueeze(0)
            .expand(numBox, numPose, 4, 4)
            .reshape(-1, 4, 4)
        )
        cameraRotatePart = cameraPosePart[:, :3, :3]
        cameraTransPart = cameraPosePart[:, :3, 3]
        bboxCentersPart = (
            bboxCentersList[i].unsqueeze(1).expand(numBox, numPose, 3).reshape(-1, 3)
        )

        cameraRotateParts.append(cameraRotatePart)
        cameraTransParts.append(cameraTransPart)
        bboxCentersParts.append(bboxCentersPart)

    cameraRotateAll = torch.cat(cameraRotateParts)

    cameraTransAll = torch.cat(cameraTransParts)

    bboxCenters = torch.cat(bboxCentersParts)

    transDist = torch.norm(cameraTransAll - bboxCenters, dim=-1)

    transVecAll = bboxCenters - cameraTransAll

    orientVecAll = cameraRotateAll[:, :3, 2]

    cossim = (
        (orientVecAll * transVecAll).sum(dim=-1)
        / torch.norm(orientVecAll, dim=-1)
        / torch.norm(transVecAll, dim=-1)
    )

    transCloseness = 1 + (1 - transDist / transDist.max())

    transCloseness[cossim <= cossimThresh] = 0
    cossim[cossim > cossimThresh] = 0

    score = cossim + transCloseness

    bboxPoseIDsList = []
    curr = 0
    for i, (numBox, numPose) in enumerate(numBoxPoseList):
        numBoxPose = numBox * numPose
        scorePart = score[curr : (curr + numBoxPose)].reshape(numBox, numPose)
        curr += numBoxPose
        _, score_topk_index = torch.topk(
            scorePart, min(numFrame, scorePart.shape[1]), dim=1
        )

        bboxPoseIDsList.append(poseIDsList[i][score_topk_index])

    return bboxPoseIDsList


def generateImageBboxPairs(scansPath, sceneID, framePath, numFrame=3):
    bboxFile = join(scansPath, sceneID, f"{sceneID}_bbox.npy")
    bboxes = np.load(bboxFile)

    sceneFramePath = join(framePath, sceneID)

    poseDir = join(sceneFramePath, "pose")
    poseFiles = [join(poseDir, poseFname) for poseFname in os.listdir(poseDir)]

    bboxCenters = []
    for bbox in bboxes:
        bboxCenters.append(bbox[:3])
    bboxCenters = [torch.from_numpy(bboxCenter) for bboxCenter in bboxCenters]
    bboxCenters = torch.stack(bboxCenters)

    cameraPoseDic = getCameraPoseDic(poseFiles)
    cameraPoseIDs = np.array(list(cameraPoseDic.keys()))
    cameraPoses = torch.from_numpy(np.stack(list(cameraPoseDic.values())))

    imagePoseIDsList = getBboxImage(
        [bboxCenters],
        [cameraPoseIDs],
        [cameraPoses],
        numFrame=numFrame,
        cossimThresh=0.9,
    )

    imageBboxPairs = []
    processedImagePaths = set()  # Set to keep track of processed image paths

    for i, imagePoseIDs in enumerate(imagePoseIDsList):
        for poseIDs in imagePoseIDs:
            for poseID in poseIDs:
                imagePath = join(sceneFramePath, "color", f"{poseID}.jpg")
                if (
                    imagePath not in processedImagePaths
                ):  # Check if the image path is already processed
                    processedImagePaths.add(imagePath)  # Add the image path to the set
                    imageBboxPairs.append((imagePath, bboxes[i]))
    imageBboxPairs = sorted(
        imageBboxPairs, key=lambda pair: int(pair[0].split(".")[0].split("/")[-1])
    )
    return imageBboxPairs


# GENERATE CAPTIONS LIST


def process_image(file, path, vis_processors, device):
    num, _ = file.split(".")
    num = num.split("/")[-1]
    rawImage = Image.open(join(path, str(num) + ".jpg"))
    rawImage = resize(rawImage, (364, 364))
    image = vis_processors["eval"](rawImage).to(device)
    return num, image


#
#
def generateCaptionBatch(files, path, model, vis_processors, device):
    images = []
    nums = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for file in files:
            future = executor.submit(process_image, file, path, vis_processors, device)
            futures.append(future)
        for future in futures:
            num, image = future.result()
            images.append(image)
            nums.append(num)
    images = torch.stack(images, dim=0)
    tempCaptions = model.generate({"image": images})
    return nums, tempCaptions


def generateCaptions(
    scansPath,
    sceneIDs,
    framePath,
    modelClass,
    modelPath,
    modelImgSize,
    vitPrecision,
    freezeVit,
    optModel,
    defaultConfig,
    batchSize=4,  # Specify the batch size here
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_cls = registry.get_model_class(modelClass)    
    if model_cls is not None:
        model = model_cls(
            img_size=modelImgSize,
            vit_precision=vitPrecision,
            freeze_vit=freezeVit,
            opt_model=optModel,
        )
    else:
        # throw exception
        raise Exception("Model class not found")
        
    model.load_checkpoint(modelPath)
    model.eval()
    cfg = OmegaConf.load(defaultConfig)
    preprocess_cfg = cfg.preprocess
    vis_processors, _ = load_preprocess(preprocess_cfg)
    model.to(device)

    for sceneID in tqdm(sceneIDs, desc="Scenes", position=0):
        imageBboxPairs = generateImageBboxPairs(scansPath, sceneID, framePath)
        captions = set()
        path = join(scansPath, sceneID, "exported", "color")

        # Process images in batches
        for i in tqdm(
            range(0, len(imageBboxPairs), batchSize), desc="Processing", position=1
        ):
            batch = imageBboxPairs[i : i + batchSize]
            imagePaths, _ = zip(*batch)
            nums, tempCaptions = generateCaptionBatch(
                imagePaths, path, model, vis_processors, device
            )
            for num, tempCaption in zip(nums, tempCaptions):
                captions.add(tempCaption)

        with open(f"captions/{sceneID}.txt", "w") as outfile:
            for caption in captions:
                outfile.write(caption + "\n")
    # release the GPU memory and delete the model
    del model
    torch.cuda.empty_cache()



def generateCaptionsChatCaptioner(
    scansPath,
    sceneIDs,
    framePath,
):
    
    tempSceneIDs = []
    for sceneID in sceneIDs:
        if not os.path.exists(f"captions/{sceneID}_chat.txt"):
            tempSceneIDs.append(sceneID)
    
    sceneIDs = tempSceneIDs
    
    blip2 = Blip2("FlanT5 XXL", device_id=0, bit8=True)
    n_rounds = 5
    n_blip2_context = 1
    print_chat = 'chat'
    question_model_tag = "gpt-3.5-turbo"

    for sceneID in tqdm(sceneIDs, desc="Scenes", position=0):
        imageBboxPairs = generateImageBboxPairs(
            scansPath, sceneID, framePath, numFrame=1
        )
        captions = set()
        path = join(scansPath, sceneID, "exported", "color")
        
        # Process images in batches
        for i in tqdm(range(0, len(imageBboxPairs)), desc="Processing", position=1):
            batch = imageBboxPairs[i : i + 1]
            imagePaths, _ = zip(*batch)
            nums = []
            for imagePath in imagePaths:
                num, _ = imagePath.split(".")
                num = num.split("/")[-1]
                nums.append(num)
                img = Image.open(join(path, str(num) + ".jpg"))
                captions.add(
                    caption_image(
                        blip2,
                        img,
                        question_model_tag,
                        n_rounds,
                        n_blip2_context,
                        print_chat,
                    )['ChatCaptioner']['caption']
                )
        # save captions to file
        with open(f"captions/{sceneID}_chat.txt", "w") as outfile:
            for caption in captions:
                outfile.write(caption + "\n")
        # create file with chatcaptioner captions interlaced with the captions from captions/sceneID.txt
        # handle that they may have different numbers of captions
        with open(f"captions/{sceneID}_chat_interlaced.txt", "w") as outfile:
            with open(f"captions/{sceneID}.txt", "r") as infile:
                captions = infile.readlines()
                with open(f"captions/{sceneID}_chat.txt", "r") as infile2:
                    captions2 = infile2.readlines()
                    for i in range(max(len(captions), len(captions2))):
                        if i < len(captions):
                            outfile.write(captions[i])
                        if i < len(captions2):
                            outfile.write(captions2[i])


# ANSWER QUESTION


# deletes all created files to start fresh
def resetTesting(sceneIDs, scansPath, rgbPath, savePath, save2dmaskPath):
    # delete segmentator
    try:
        os.remove(join("Segmentator", "segmentator"))
    except:
        pass

    # delete targets/sceneIDs/sceneIDs_vh_clean_2.0.010000.segs.json
    for sceneID in sceneIDs:
        try:
            os.remove(
                join(scansPath, sceneID, f"{sceneID}_vh_clean_2.0.010000.segs.json")
            )
        except:
            pass
        try:
            os.remove(
                join(scansPath, sceneID, "targets", f"{sceneID}_vh_clean_2.labels.ply")
            )
        except:
            pass

        # delete rgbPath/sceneID directory
        try:
            shutil.rmtree(join(rgbPath, sceneID))
        except:
            pass

        # delete SavePath/sceneID.pth
        try:
            os.remove(join(savePath, sceneID + ".pth"))
        except:
            pass

        # delete Save2dMaskPath/sceneID directory
        try:
            shutil.rmtree(join(save2dmaskPath, sceneID))
        except:
            pass
        # delete sceneID.txt
        try:
            os.remove(join(sceneID + ".txt"))
        except:
            pass


def main():
    import constants

    # createSegmentatorExecutable()
    # performScannetPreprocessing(constants.ScanNetPath, constants.DataPath)
    # perform2DDataPreprocessing(
    #    constants.ScansPath, constants.RGBPath, constants.Width, constants.Height
    # )
    # perform3DSegmentationNoMultiprocessing(
    #    constants.TrainPath,
    #    constants.ValPath,
    #    constants.CheckpointPath,
    #    constants.SavePath,
    #    constants.RGBPath,
    #    constants.DataPath,
    #    constants.Save2DMaskPath,
    #    constants.VoxelSize,
    #    constants.Threshold,
    # )
    # generateBboxEnsembles(
    #    constants.ScansPath,
    #    constants.SceneIDs,
    #    constants.TrainPath,
    #    constants.ValPath,
    #    constants.SavePath,
    #    constants.PreProcessPath,
    # )
    # generateCaptions(
    #     constants.ScansPath,
    #     constants.SceneIDs,
    #     constants.ScanReferFramesPath,
    #     constants.ModelClass,
    #     constants.ModelPath,
    #     constants.ModelImgSize,
    #     constants.VitPrecision,
    #     constants.FreezeVit,
    #     constants.OptModel,
    #     constants.DefaultModelConfig,
    # )
    
    generateCaptionsChatCaptioner(
        constants.ScansPath,
        constants.SceneIDs,
        constants.ScanReferFramesPath,
    )


if __name__ == "__main__":
    main()
