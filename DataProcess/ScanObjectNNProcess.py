

import h5py
import os


split='1'
file_path = "./s"+split+'/test_objectdataset.h5'  # Replace with the actual file path

if not os.path.exists(split):
    os.makedirs(split)

if 'test' in file_path:
    app='t_'
else:
    app=''

# Open the HDF5 file
hdf5_file = h5py.File(file_path, 'r')

# Function to recursively print the structure of the HDF5 file
def print_hdf5_structure(item, indent=""):
    if isinstance(item, h5py.Group):
        print(f"{indent}Group: {item.name}")
        for key in item.keys():
            sub_item = item[key]
            print_hdf5_structure(sub_item, indent + "  ")
    elif isinstance(item, h5py.Dataset):
        print(f"{indent}Dataset: {item.name} (Shape: {item.shape}, Dtype: {item.dtype})")

# Call the function to print the HDF5 structure
print_hdf5_structure(hdf5_file)

# Close the file

# Access the 'data' and 'label' datasets
data_dataset = hdf5_file['data']
label_dataset = hdf5_file['label']
data_array = data_dataset[:]
label_array = label_dataset[:]

#print("Data:")
#print(data_array[:3])  # Print the first three samples
#print("Label:")
#print(label_array[:3])  # Print the corresponding labels

print(data_array.shape)

exit()
counter=0

for i,label in enumerate(label_array):
    write_data=data_array[i, :, :]
    folder_name="./"+split+"/"+str(label)    
    import os
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        counter=1
    else:
         counter=counter+1   
    filename=folder_name+"/"+app+str(label)+"_"+"{:06d}".format(counter)+".txt"
    print(filename)
    with open(filename, 'w') as csvfile:
        for row in write_data:
            csv_line = ','.join(map(str, row))
            csvfile.write(csv_line + '\n')

hdf5_file.close()


#Actual clasnames
https://github.com/hkust-vgd/scanobjectnn/issues/38
https://github.com/hkust-vgd/scanobjectnn/blob/master/training_data/shape_names_ext.txt
