import argparse
import albumentations as A 
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from dataset import download_dataset, get_dataloaders
from train import train_validation
from inference import inference
import vis_utils #my visualize model


def run(args): 

    train_tfs = A.Compose([A.Resize(args.image_size, args.image_size), A.VerticalFlip(0.5), A.HorizontalFlip(0.5), A.GaussNoise(0.2)])

    assert args.down_path == "data", "Iltimos data papkasini kiriting"
    # 1 - download dataset
    root_ds = download_dataset(path_to_download = args.down_path, dataset_name = args.dataset_name)
    # 2 - get dataloaders 
    tr_dl, val_dl, test_dl = get_dataloaders(dataset_path=root_ds, tfs=train_tfs, bs=args.batch_size)

    # 3 - save visualized examples from dataset (not dataloaders)
    for dl, data_type in zip([tr_dl, val_dl, test_dl], ['train', 'val', 'test']):
        vis_utils.visualize_ds(dataset=dl.dataset, num_images=args.n_imgs, data_type=data_type, save_folder=args.vis_path)

    # 4 - train and validation process
    model = args.model_name(encoder_name=args.model_backbone, encoder_depth=5, 
                                encoder_weights="imagenet", decoder_channels=256, in_channels=3, 
                                classes = args.n_classes, activation = None, upsampling=8, aux_params=None)
    # model = smp.DeepLabV3(encoder_name=args.b, encoder_depth=5, 
    #                         encoder_weights="imagenet", decoder_channels=256, in_channels=3, 
    #                         classes = args.n_classes, activation = None, upsampling=8, aux_params=None)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    
    print('...................TRAIN JARAYONI BOSHLANDI!.........................\n')
    #start training and validation process and get acc and loss results as list
    model = model.to(args.device)
    model.train()
    #after train and validation function return tr and val loss, pa, and mIOU as dictionary
    result_dict = train_validation(model=model, tr_dl=tr_dl, val_dl=val_dl, n_cls=args.n_classes, 
                                   loss_fn=loss_fn, optim=optimizer, epochs=args.epochs, device=args.device, 
                                   save_dir=args.model_files, save_prefix=args.dataset_name)

    print('...................TRAIN JARAYONI YAKUNLANDI!.........................\n')
    # 5 - Save training and validation loss, PA and mIOU metric plots to folder
    vis_utils.visualize_seg_metrics(result=result_dict, save_dir=args.vis_path)

    #6 - Inference part 
    print('...................INFERENCE JARAYONI BOSHLANDI!......................\n')
    model.load_state_dict(torch.load(f"{args.model_files}/{args.dataset_name}_best_model.pth"))
    model.eval()
    #this function execute inference and return inference data for plotting 
    inference_data = inference(model=model, test_dl=test_dl, loss_fn=loss_fn, device=args.device)

    #Save inference result images to args.vis_path folder
    vis_utils.visualize_inference(inference_data, len(test_dl), args.n_imgs, args.vis_path)
    print('...................INFERENCE JARAYONI YAKUNLANDI!......................\n')


if __name__ == "__main__": 
  
    parser = argparse.ArgumentParser(description="Insect binary segmentation")
    parser.add_argument("-dp", "--down_path", type=str, default="data", help="Datasetni yuklash uchun path")
    parser.add_argument("-dn", "--dataset_name", type=str, default = "arthropodia", help="Dataset nomi")
    parser.add_argument("-n_im", "--n_imgs", type=int, default = 20, help="Number of images for plotting")
    parser.add_argument("-vs", "--vis_path", type=str, default="data/plots", help="Vizualizations graph, plotlarni saqlash uchun yo'lak")
    parser.add_argument("-mn", "--model_name", type=smp.DeepLabV3, default = smp.DeepLabV3, help="Pretrained bo'lgan pytorch segmentation model nomi")
    parser.add_argument("-mb", "--model_backbone", type=str, default="resnet34", help="Pytorch segmentation model uchun backbone arxitektura nomi")
    parser.add_argument("-nc", "--n_classes", type=int, default=2, help="Number of segmentation classes")    
    parser.add_argument("-ims", "--image_size", type=int, default=320, help="Image size to be resized")  
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="Epochlar soni")
    parser.add_argument("-bs", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-dv", "--device", type=str, default="cuda:0", help="Train qilish qurilmasi GPU yoki CPU")
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", 
                        help="Train qilingan model va boshqa parametr fayllar uchun yo'lak")
    args = parser.parse_args()
    run(args)