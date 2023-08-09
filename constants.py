import json

ScansPath = "data/sqa3dTest"
# auto create sqa3dTestScenes.txt based on the scenes in the ScansPath
# import os
# with open("sceneIDLists/sqa3dTestScenes.txt", "w") as f:
#     for scan in os.listdir(ScansPath):
#         # make sure it is a scan folder (starts with scene)
#         if scan[0:5] == "scene":
#             f.write(scan + "\n")


# SCENE IDS

# \n seperated list of scene IDs to evaluate model with
evaluationSceneList = "sceneIDLists/sqa3dTestScenes.txt"
SceneIDs = []
with open(evaluationSceneList, "r") as f:
    for line in f:
        SceneIDs.append(line.strip())

# PATHS - CHANGE THESE TO YOUR OWN IF NEEDED
RGBPath = "data/scannetv2_images"
DataPath = "data/scannet_3d"
SavePath = "data/savepcd"
Save2DMaskPath = "data/save2dmask"
TrainPath = "sceneIDLists/scannetv2_train.txt"
ValPath = "sceneIDLists/scannetv2_val.txt"
PreProcessPath= "data/preprocess_output"
ScanNetPath = "data/ScanNet"
ScanReferFramesPath = "data/frames_square"

# SAM3D
CheckpointPath = "models/SAM.pth"
Width = 640
Height = 480
ImageSize = [640, 480]
VoxelSize = 0.05
Threshold = 50

# FRAME GENERATION
NumFrames = 10
CossimThresh = 0.9

# CAPTION GENERATION
ModelClass = "blip2_opt"
ModelPath = "models/location_captions.pth"

# LAVIS SETTINGS
ModelImgSize = 364
VitPrecision = "fp32"
FreezeVit = True
OptModel = "facebook/opt-6.7b"
DefaultModelConfig = "models/finetunedConfig.yaml"
BatchSize = 4

# QA

# Questions
# Read from v1_valanced_questions_test_scannetv2.json
QuestionsPath = "v1_balanced_questions_test_scannetv2.json"
Questions = []
with open(QuestionsPath, "r") as f:
    Questions = json.load(f)
Questions = Questions["questions"]

# Answers
# Read from v1_valanced_annnotations_test_scannetv2.json
AnswersPath = "v1_balanced_sqa_annotations_test_scannetv2.json"
Answers = []
with open(AnswersPath, "r") as f:
    Answers = json.load(f)
Answers = Answers["annotations"]