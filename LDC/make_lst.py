import os
import natsort

TRAIN_ORG = r"data/np_train/org"
TRAIN_ED = r"data/np_train/edge"

TEST_ORG = r"data/np_val/org"
TEST_ED = r"data/np_val/edge"

VAL_ORG = r"data/np_val/org"
VAL_ED = r"data/np_val/edge"

SAVE_PATH = r"data/np_train"

train_org_list = natsort.natsorted(os.listdir(TRAIN_ORG))
train_ed_list = natsort.natsorted(os.listdir(TRAIN_ED))

test_org_list = natsort.natsorted(os.listdir(TEST_ORG))
test_ed_list = natsort.natsorted(os.listdir(TEST_ED))

val_org_list = natsort.natsorted(os.listdir(VAL_ORG))
val_ed_list = natsort.natsorted(os.listdir(VAL_ED))

for cur in ["train","test","val"]:
    if cur == "train":
        cur_org_list = train_org_list
        cur_ed_list = train_ed_list
        org_path = TRAIN_ORG
        ed_path = TRAIN_ED
    elif cur == "test":
        cur_org_list = test_org_list
        cur_ed_list = test_ed_list
        org_path = TEST_ORG
        ed_path = TEST_ED
    else:
        cur_org_list = val_org_list
        cur_ed_list = val_ed_list
        org_path = VAL_ORG
        ed_path = VAL_ED

        
    for org,edge in zip(cur_org_list,cur_ed_list):
        full = "org"+"/"+org+" "+"edge"+"/"+edge
        with open(SAVE_PATH+"/"+f"{cur}_pair.lst", "a") as a_file:
                    a_file.write(full)
                    a_file.write("\n")