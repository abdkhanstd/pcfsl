import os
import json

# Following the list provided supplementary material in your CVPR 2022 paper

dataset_dir = "process/ShapeNetCore"

#train_categories = sorted(["chair", "sofa", "airplane", "bed", "monitor", "table", "toilet", "mantel", "tv stand", "plant", "car", "desk", "dresser", "glass box", "guitar", "bench", "cone", "tent", "laptop", "curtain", "radio", "xbox", "bathtub", "lamp", "stairs", "door", "stool", "wardrobe", "cup", "bowl"])
#test_categories = sorted(["bookshelf", "vase", "bottle", "piano", "night stand", "range hood", "flower pot", "keyboard", "sink", "person"])

train_categories = ["sofa", "desk", "phone", "armchair", "bus", "bathtub", "radiotelephone", "park_bench",
                         "coffee_table", "club_chair", "boat", "sniper_rifle", "clock", "pot", "jar", "dresser",
                         "table_lamp", "laptop", "sedan", "knife", "rectangular_table", "coupe", "swivel_chair",
                         "desk_cabinet", "motorcycle", "race_car", "garage_cabinet", "handgun", "sport_utility", "piano",
                         "bed", "stove", "convertible", "sports_car", "bowl", "cruiser", "carbine", "printer", "microwave",
                         "skateboard", "tower", "cantilever_chair", "basket", "beach_wagon", "can", "pillow", "jeep",
                         "dishwasher", "rocket", "bag"]

test_categories=["display","airline","guitar","faucet","jet","fighter","bottle","bookshelf","train","ashcan","file_cabinet","swept_wing","mug","washer","helmet","propeller_plane","bomber","delta_wing","camera","mailbox"];


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
