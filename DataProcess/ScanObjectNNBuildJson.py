import os
import json

# Following the list provided supplementary material in your CVPR 2022 paper

dataset_dir = "/home/abdkhan/myfsl/dataset/ScanObjectL2"

train_categories_actual = sorted(["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed", "pillow", 
                           "sink", "sofa", "toilet"])
#Split 1
test_categories = sorted(["shelf", "door", "bin", "box", "bag"])

#Split 2
#test_categories = sorted(["chair", "sofa", "desk", "bed", "pillow"])

#Split 3
#test_categories = sorted(["cabinet", "table", "display", "sink", "toilet"])

# Rebuild
train_categories=[]
for cat in train_categories_actual:
    if cat not in test_categories:
        train_categories.append(cat)
        
print(train_categories)        


label_num = 0
count = 1

# First handling the training set
tot_train = 0
tot_test = 0
print("Cat\t\tItems")

lbl_ctr = 1
data = {"label_names": sorted(train_categories + test_categories), "image_names": [], "image_labels": []}

for category in train_categories:
    dir_name = os.path.join(dataset_dir, category)
    if os.path.exists(dir_name):
        listing = os.listdir(dir_name)
    else:
        listing=0


    if listing:

        # saving image names and assigning label
        print(f"{category}\t\t{len(listing)}")
        tot_train += len(listing)
        for filename in listing:
            data["image_names"].append(os.path.join(".", category, filename))
            data["image_labels"].append(label_num)
            count += 1

        label_num += 1

print("Total samples:", tot_train)

# Write it to JSON file
with open('base.json', 'w') as file:
    json.dump(data, file)

label_num = 0
count = 1

data2 = {"label_names": sorted(train_categories + test_categories), "image_names": [], "image_labels": []}
for category in test_categories:
    dir_name = os.path.join(dataset_dir, category)
    if os.path.exists(dir_name):
        listing = os.listdir(dir_name)
    else:
        listing=0


    if listing:
        # saving image names and assigning label
        print(f"{category}\t\t{len(listing)}")
        tot_test += len(listing)
        for filename in listing:
            data2["image_names"].append(os.path.join(".", category, filename))
            data2["image_labels"].append(label_num)
            count += 1

        label_num += 1

print("Total samples:", tot_test)

# Write it to JSON file
with open('novel.json', 'w') as file:
    json.dump(data2, file)
