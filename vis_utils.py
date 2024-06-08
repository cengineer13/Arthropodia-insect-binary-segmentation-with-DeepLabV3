import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def get_random_data(dataset:str, num_images:int):
 #""" This function services to get random data points from main dataset and return data """
    data = []
    for i in range(num_images):
        random_index = np.random.randint(1, len(dataset))
        tensor_img, tensor_mask = dataset[random_index] #get random index image and label from dataset
        np_image = ((tensor_img)*255).cpu().permute(1,2,0).numpy().astype("uint8") #tensor to np
        np_mask = tensor_mask.cpu().numpy().astype('uint8')
        data.append((np_image, np_mask, random_index)) #save as tuple

    return data

def visualize_ds(dataset, num_images, data_type, save_folder):

    plot_data = get_random_data(dataset, num_images)
    # Create figure and subplots (side-by-side in this example)
    fig, axes = plt.subplots(len(plot_data), 2, figsize=(5, len(plot_data)*2))  # Adjust figsize as needed 
    # Loop through image and labels and plot
    for i in range(len(plot_data)):
        axes[i, 0].imshow(plot_data[i][0])
        axes[i, 1].imshow(plot_data[i][1], cmap='gray')
        axes[i, 0].set_title(f"Orginal Images {plot_data[i][2]}")
        axes[i, 1].set_title(f"Label Masks {plot_data[i][2]} ")
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        plt.tight_layout()

    #Make a dir for saving plots 
    os.makedirs(f"{save_folder}", exist_ok=True)
    plt.savefig(f"{save_folder}/1-{data_type} image and mask random examples.png")
    print(f"{data_type} datasetdan namunalar {save_folder} papkasiga yuklandi...")
    print("---------------------------------------------------------------------")
    plt.clf(); plt.close()
    #plt.show()


def visualize_seg_metrics(result:dict, save_dir):

    plt.figure(figsize=(8,4))
    plt.plot(result["tr_loss"], label = "Train loss")
    plt.plot(result["val_loss"], label = "Validation loss")
    plt.title("Train and Validation Loss")
    plt.xticks(np.arange(len(result["val_loss"])), [i for i in range(1, len(result["val_loss"])+1)])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(-0.5, 3)
    plt.legend()
    
    plt.savefig(f"{save_dir}/2-Training and Validation loss metrics.png")
    print(f" Training and Validation loss metricslar {save_dir} papkasiga yuklandi...\n")
    #plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(result["tr_pa"], label = "Train PA")
    plt.plot(result["val_pa"], label = "Validation PA")
    plt.title("Train and Validation PA ")
    plt.xticks(np.arange(len(result["val_pa"])), [i for i in range(1, len(result["val_pa"])+1)])
    plt.xlabel("Epochs")
    plt.ylabel("PA score")
    plt.legend()
    plt.ylim(-0.5, 3)
    plt.savefig(f"{save_dir}/2-Training and Validation PA metrics.png")
    print(f" Training and Validation PA pixel accuracy metrics {save_dir} papkasiga yuklandi...\n")
    #plt.show()

    
    plt.figure(figsize=(8,4))
    plt.plot(result["tr_iou"], label = "Train mioU")
    plt.plot(result["val_iou"], label = "Validation mioU")
    plt.title("Train and Validation mIOU ")
    plt.xticks(np.arange(len(result["val_iou"])), [i for i in range(1, len(result["val_iou"])+1)])
    plt.xlabel("epochs")
    plt.ylabel("mIoU score")
    plt.legend()
    plt.savefig(f"{save_dir}/2-Training and Validation mIOU metrics.png")
    print(f" Training and Validation mIOU metricslar {save_dir} papkasiga yuklandi...\n")
    #plt.show()


def visualize_inference(inference_data, len_test_dl, n_imgs, save_dir):
    #Create figure and subplots 
    fig, axes = plt.subplots(n_imgs, 3, figsize=(10, n_imgs*2))  
    for i in range(n_imgs):
        random_idx = np.random.randint(0, len_test_dl)
        axes[i, 0].imshow(inference_data[random_idx][0]) #0 - org img,  2-pred mask
        axes[i, 1].imshow(inference_data[random_idx][1], cmap='gray') #1-mask
        axes[i, 2].imshow(inference_data[random_idx][2], cmap='gray') #3-pred mask
        axes[i, 0].set_title(f"{random_idx} - Orginal image")
        axes[i, 1].set_title(f"{random_idx} - Label mask")
        axes[i,2].set_title(f"{random_idx} - Prediction mask")
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
        axes[i, 2].axis('off')
        plt.tight_layout()
    plt.savefig(f"{save_dir}/3-Inference_result_examples.png")
    print(f"Inferencedan keyingi natija random example rasmlar {save_dir} papkasiga yuklandi...")
    #plt.show()