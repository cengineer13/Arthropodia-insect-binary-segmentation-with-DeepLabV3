import argparse, cv2
import torch
import albumentations as A
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms as T
import streamlit as st
import matplotlib.pyplot as plt

def run(args): 

    test_tfs = T.Compose([T.ToTensor(), T.Resize((args.image_size, args.image_size))])

    #load our best model
    model = load_model(model_name=args.model_name, model_backbone=args.model_backbone, model_path=args.model_files, 
                       dataset_name=args.dataset_name, num_classes=args.n_classes)
    
    print(f"Train qilingan model {args.model_name} muvaffaqiyatli yuklab olindi.!")

    #img for prediction
    st.title("Cloud quality Classification python project")
    img_path = st.file_uploader("Rasmni yuklang...")

                        #1-parm - model, 2-img_path, 3-transformations, 4-class_names 
    image, mask = predict(model, img_path, test_tfs, args.device) if img_path else predict(model, args.test_img, test_tfs, args.device)

    # Create two columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Orginal image")
        st.write("Bashorat uchun rasm")

    with col2:
        st.image(mask, caption="Predicted mask image")
        st.write("Bashoratdan keyingi rasm")

def load_model(model_name, model_backbone, model_path, dataset_name, num_classes):
    
    m = model_name(encoder_name=model_backbone, encoder_depth=5, 
                            encoder_weights="imagenet", decoder_channels=256, in_channels=3, 
                            classes = num_classes, activation = None, upsampling=8, aux_params=None)
    m.load_state_dict(torch.load(f"{model_path}/{dataset_name}_best_model.pth"))
    return m.eval()

def predict(model, img_path, tfs, device):    
    #predict 
    img = Image.open(img_path).convert("RGB")
    tensor_img = tfs(img).unsqueeze(0) #3D img -> 4D as batch for model inputting
    # tensor_img = tensor_img.to(device)
    with torch.no_grad():
        prediction = model(tensor_img)
        pred_mask = torch.argmax(prediction, dim=1) #get mask prediction image
        np_pred_mask = torch.squeeze(pred_mask, dim=0).cpu().numpy()

        return img.resize((320,320), Image.BICUBIC), np_pred_mask * 255 


if __name__ == "__main__":
    #Parser classdan obyekt olamiz 
    parser = argparse.ArgumentParser(description='Arthropodia insects segmentation python project DEMO')
    
    #Add arguments (- va -- option value oladigon, type - qaysi turni olish )
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", 
                        help="Train qilingan model uchun yo'lak")
    parser.add_argument("-dn", "--dataset_name", type=str, default = "arthropodia", help="Dataset nomi")
    parser.add_argument("-ims", "--image_size", type=int, default=320, help="Image size to be resized")  
    parser.add_argument("-mn", "--model_name", type=smp.DeepLabV3, default = smp.DeepLabV3, help="Pretrained bo'lgan pytorch segmentation model nomi")
    parser.add_argument("-mb", "--model_backbone", type=str, default="resnet34", help="Pytorch segmentation model uchun backbone arxitektura nomi")
    parser.add_argument("-nc", "--n_classes", type=int, default=2, help="Number of segmentation classes") 
    parser.add_argument("-dv", "--device", type=str, default="cuda:0", help="GPU yoki CPU")
    parser.add_argument("-ti", "--test_img", default="test_images/Damsel-bug_Winfield-Sterling.jpg", help="Path for image to predict unseen image")

    #argumentlarni tasdiqlash parse
    args = parser.parse_args()

    run(args=args)