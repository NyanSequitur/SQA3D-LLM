# opens v1_balanced_questions_test_scannetv2.json and generates a .txt file of all unique sceneIDs, sorted by 
# the name, which is formatted as "scene{ID}.txt"

import json
import tqdm
split = "val"

with open(f"v1_balanced_questions_{split}_scannetv2.json") as f:
    questions = json.load(f)["questions"]
    
sceneIDs = set()
for question in tqdm.tqdm(questions):
    sceneIDs.add(question["scene_id"])
    
sceneIDs = list(sceneIDs)
sceneIDs.sort()
with open(f"{split}SceneIDs.txt", "w") as f:
    for sceneID in sceneIDs:
        f.write(sceneID + "\n")


def allSceneIDs():
    splits = ["val", "test", "train"]
    # cap number of scenes to 100
    sceneIDs = set()
    for split in splits:
        with open(f"v1_balanced_questions_{split}_scannetv2.json") as f:
            questions = json.load(f)["questions"]
            
        for question in tqdm.tqdm(questions):
            sceneIDs.add(question["scene_id"])
            
    
    sceneIDs = list(sceneIDs)
    sceneIDs.sort()
    # only use last 100 scene IDs contained within scannetv2_test.txt or scannetv2_val.txt
    
    test = []
    with open("scannetv2_test.txt") as f:
        test = f.readlines()
    test = [x.strip() for x in test]
    
    val = []
    with open("scannetv2_val.txt") as f:
        val = f.readlines()
    val = [x.strip() for x in val]
    
    sceneIDs = [x for x in sceneIDs if x in test or x in val]
    sceneIDs = sceneIDs[-100:]
    
    with open(f"allSceneIDs.txt", "w") as f:
        for sceneID in sceneIDs:
            f.write(sceneID + "\n")
allSceneIDs()