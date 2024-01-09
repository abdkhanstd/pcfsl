# Process  ShapeNetCore.V2 according to the classes given in CVPR paper
import os
import json
import numpy as np
from pathlib import Path


def find_unique_indices(arr):
    unique_indices = []  # Initialize an empty list to store unique element indices

    for index, value in enumerate(arr):
        if value not in arr[:index]:  # Check if the value occurs before the current index
            unique_indices.append(index)

    return unique_indices


train_categories_name = ["sofa", "desk", "phone", "armchair", "bus", "bathtub", "radiotelephone", "park bench",
                         "coffee table", "club chair", "boat", "sniper rifle", "clock", "pot", "jar", "dresser",
                         "table lamp", "laptop", "sedan", "knife", "rectangular table", "coupe", "swivel chair",
                         "desk cabinet", "motorcycle", "race car", "garage cabinet", "handgun", "sport utility", "piano",
                         "bed", "stove", "convertible", "sports car", "bowl", "cruiser", "carbine", "printer", "microwave",
                         "skateboard", "tower", "cantilever chair", "basket", "beach wagon", "can", "pillow", "jeep",
                         "dishwasher", "rocket", "bag"]
train_categories = ["04256520", "03179701", "04401088", "02738535", "02924116", "02808440", "02992529", "03891251",
                    "03063968", "20000027", "02858304", "04250224", "03046257", "03991062", "03593526", "03237340",
                    "04380533", "03642806", "04166281", "03624134", "20000037", "03119396", "04373704", "20000010",
                    "03790512", "04037443", "20000011", "03948459", "04285965", "03928116", "02818832", "04330267",
                    "03100240", "04285008", "02880940", "03141065", "02961451", "04004475", "03761084", "04225987",
                    "04460130", "20000020", "02801938", "02814533", "02946921", "03938244", "03594945", "03207941",
                    "04099429", "02773838"]


test_categories_names=["display","airline","guitar","faucet","jet","fighter","bottle","bookshelf","train","ashcan","file cabinet"," swept wing","mug","washer","helmet","propeller plane","bomber","delta wing","camera","mailbox"];
test_categories=["03211117","02690373","03467517","03325088","03595860","03335030","02876657","02871439","04468005","02747177","03337140","20000001","03797390","04554684","03513137","04012084","02867715","03174079","02942699","03710193"];


'''
# Thers a problem with the cateagories mentioned in  Ye et al paper. So, im managing the missing caetagories seprately
# Comment and use apropriately
train_categories_name=['bed','bed','bed','bed','bed','bed','bed','bed','bed','bed','bed','bed','bed'];
train_categories=['02831724','02920083','02920259','03114504','03115762','03388549','03482252','03962852','04222210','20000004','20000005','20000006','20000007'];

train_categories_name=[ "tower", "tower", "tower", "tower", "tower", "tower", "tower", "tower", "tower", "tower", "tower", "tower"]
train_categories=[ "02814860", "02826886", "03029197", "03047052", "03519387", "04028581", "04206790", "04220250", "04312432", "04361260", "04501947", "04556948"]

# This class is anbigiuis as the calss cant be found in v2 or v1 dataset and has no cildren
test_categories_names=['swept wing']
test_categories=['20000001']

train_categories_name=[]
train_categories=[]

# handeling classes with lesser
'''


# Merging for now
train_categories_name_m=train_categories_name+test_categories_names
train_categories_m=train_categories+test_categories

additiona_cat_names=['bed','tower','']
additiona_cat=['02818832','04460130','']

#Defining additional
train_categories_name_m_2=[]
train_categories_m_2=[]
# There are some problems with vague decription, so well have to add some childens to balcne the numbers
#Pre-process the array,as it seems that the authors have added childs aswell
with open('taxonomy.json', 'r') as json_file:
    data = json.load(json_file)    

for index, id in enumerate(additiona_cat):
   
    for index_row, row in enumerate(data):
        if row['synsetId']==id:
            if row['children']:
                #Append the original name and childs aswell
                #Append the name and child id to row
                for _,nums in enumerate(row['children']):
                    train_categories_m_2.append(nums)
                    train_categories_name_m_2.append(additiona_cat_names[index])
            
print(len(train_categories_name_m))
train_categories_name_m=train_categories_name_m+train_categories_name_m_2
train_categories_m=train_categories_m+train_categories_m_2       

'''
#Removing Duplicates synids now
index=find_unique_indices(train_categories_m)
train_categories_name_m = [train_categories_name_m[index] for index in index]
train_categories_m = [train_categories_m[index] for index in index]

print(train_categories_m)
'''

# read and store in memory the taxonomy file  values = []
values = []
num_lines=0
with open('mixed_taxonomy.txt', "r") as f:
    for line in f: #loop over file
      num_lines=num_lines+1
      strips = line.strip().split(",")
      id = strips[0]
      for index, wid in enumerate(train_categories_m):
        try:          
            if wid in line:
              try:          
                  class_name=train_categories_name_m[index]         
                  values.append((id, class_name ))     
              except ValueError:
                pass                        
        except ValueError:
          pass            
     

print("Found from mixed taxonomy",len(values),num_lines) 
exit()
# Create a dictionary to keep track of unique 'id' and 'class_name' pairs
unique_values_dict = {}
for id_val, class_name in values:
    if id_val not in unique_values_dict:
        unique_values_dict[id_val] = class_name

# Convert the dictionary back to a list of tuples (id, class_name) without duplicates
values = list(unique_values_dict.items())



def save_last_id(digit):
    with open("line.txt", 'w') as file:
        file.write(str(digit))

def search_folder(directory, folder_name):
    for root, dirs, files in os.walk(directory):
        if root.endswith(os.path.sep + folder_name):
            return os.path.join(directory, root)

def read_obj_file(file_path):
    vertices = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):  # Lines starting with 'v ' represent vertices
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)       
    return vertices 
  
def create_folder_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)  
        
def write_vertices_to_file(file_path, vertices):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            x, y, z = vertex
            file.write(f"{x:.6f},{y:.6f},{z:.6f}\n")
            
def read_id_from_file():
    with open("line.txt", 'r') as file:
        line = file.readline()
        return line
            

### List of operations is generated till here, now start copying by making folders
min_points=2048
dataset_path='process/ShapeNetCore.v2/'
target_dataset_path='process/ShapeNetCore.abd'
create_folder_if_not_exist(target_dataset_path)  

counter = np.zeros(len(train_categories_name_m)).astype(int)

cnt=0
lv=len(values)

if_continue=0 # Zero means dont continue
last_id=read_id_from_file()
cflag=if_continue+1

for row in values:
  
  if if_continue==1 and last_id==row[0]:
    cflag=1
  elif cflag !=1:      
    cnt=cnt+1

    
  if cflag==1:      
    given_folder=row[0]
    Obj_file_name = search_folder(dataset_path,given_folder)+'/models/model_normalized.obj'
    
    if Path(Obj_file_name).is_file(): # checkif file exist as some files are missing in the original dataset
        vertices= read_obj_file(Obj_file_name)
        if len(vertices) >= min_points:
            # Get the index number for counter that creates names
            index=train_categories_name_m.index(row[1])
            
            # Save the file in current folder
            new_folder_path=target_dataset_path+'/'+row[1];  
            new_file_name=new_folder_path+'/'+row[1]+"_{:06d}.txt".format(counter[index])
            new_file_name=new_file_name.replace(' ','_')
            new_folder_path=new_folder_path.replace(' ','_')
            print("File Number: ",cnt,"/",lv, "  ->",new_file_name)
            

            
            # create folder
            create_folder_if_not_exist(new_folder_path) 
            
            ## update counter 
            counter[index] = counter[index]+1 
            
            ## Write vertices
            write_vertices_to_file(new_file_name,vertices)
            
            save_last_id(row[0])
            cnt=cnt+1

    
