import os, shutil, glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.utils.data import random_split
torch.manual_seed(2024)
# a)datasetni yuklab olish
def download_dataset(path_to_download, dataset_name = "arthropodia"): 
    
    assert dataset_name == "arthropodia", f"Iltimos arthropodia nomi bilan kiriting!"
    if dataset_name == "arthropodia": url = "kaggle datasets download -d killa92/arthropodia-semantic-segmentation-dataset"
    
    # Check if is already exist 
    if os.path.isfile(f"{path_to_download}/{dataset_name}.csv") or os.path.isdir(f"{path_to_download}/{dataset_name}"): 
        print(f"Dataset allaqachon yuklab olingan. {path_to_download}/{dataset_name} papkasini ni tekshiring."); 

    # If data doesn't exist in particular folder
    else: 
        ds_name = url.split("/")[-1] 
        # Download the dataset
        print(f"{ds_name} yuklanmoqda...")
        os.system(f"{url} -p {path_to_download}")
        shutil.unpack_archive(f"{path_to_download}/{ds_name}.zip", extract_dir=f"{path_to_download}/{dataset_name}")
        os.remove(f"{path_to_download}/{ds_name}.zip")
        print(f"Tanlangan dataset {path_to_download}/{dataset_name} papkasiga yuklab olindi!")
    
    return f"{path_to_download}/{dataset_name}"
    

#make a Custom dataset class 
class ArthropodiaDataset(Dataset): 
    def __init__(self, dataset_path, transformations = None):
        super().__init__()
        self.transformations = transformations
        self.tensorize = T.Compose([T.ToTensor()])
        self.image_paths = sorted(glob.glob(f"{dataset_path}/*/images/*"))
        self.label_paths = sorted(glob.glob(f"{dataset_path}/*/labels/*"))

        # print(self.image_paths)
        # print(self.label_paths)
    
    def __len__(self):  return len(self.image_paths)

    def __getitem__(self, index): 

        image = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.label_paths[index]), cv2.COLOR_BGR2GRAY)

        if self.transformations is not None: 
            transformed = self.transformations(image = image, mask = mask)
            image = transformed['image']
            mask = transformed['mask'] 

        image = self.tensorize(image)
        mask = torch.tensor(mask>128).long()
        
        return image, mask 


def get_dataloaders(dataset_path, tfs, bs, split = [0.9, 0.05, 0.05]):
    
    cloud_ds = ArthropodiaDataset(dataset_path=dataset_path, transformations=tfs)
    train_data, valid_data, test_data  = random_split(dataset = cloud_ds, lengths = split)

    print(f"Train data size:{len(train_data)}   |  Valid data size:{len(valid_data)}    |    Test data size: {len(test_data)}\n")

    train_dataloader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True,)
    val_dataloader = DataLoader(dataset = valid_data, batch_size=bs,shuffle=False, )
    test_dataloader = DataLoader(dataset = test_data, batch_size=bs,  shuffle=False, )

    return train_dataloader, val_dataloader, test_dataloader
