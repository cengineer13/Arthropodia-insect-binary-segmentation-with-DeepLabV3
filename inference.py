import torch
import numpy as np
from metrics import Metrics

def inference(model, test_dl, loss_fn, device): 

    inference_data = [] #store images for ploting
    test_loss, test_pa, test_iou = 0, 0, 0
    for images, masks in test_dl:
        t_los, t_pa, t_iou = [],[],[] #loss, pa, and iou per image   
        for image, mask in zip(images, masks): 

            with torch.no_grad():
                image, mask = image.to(device), mask.to(device)
                prediction = model(torch.unsqueeze(image, dim=0))
                pred_mask = torch.argmax(prediction, dim=1) 
                
            #Convert tensors to Numpy arrays for plotting and store data to plot
            np_image = image.cpu().numpy().transpose(2,1,0)
            np_mask = mask.cpu().numpy()
            np_pred_mask = torch.squeeze(pred_mask, dim=0).cpu().numpy()
            inference_data.append((np_image, np_mask, np_pred_mask)) #add data to the list as tuple

            #get metrics in each batch per loop
            met = Metrics(pred=prediction, gt=torch.unsqueeze(mask,dim=0), loss_fn=loss_fn, n_cls=2)
            losses = met.loss()
            
            #batch ichidagi har bir rasm loss, pixel acc va iou larini hisoblaydi. 
            t_los.append(losses.item())
            t_pa.append(met.PA())
            t_iou.append(met.mIoU())

        test_loss += np.mean(t_los)
        test_pa += np.mean(t_pa)
        test_iou += np.mean(t_iou)

    #INFERENCE natijasi 
    test_loss /= len(test_dl)
    test_pa /= len(test_dl)
    test_iou /= len(test_dl)

    print(f"INFERENCE NATIJASI:")
    print(f"Test Loss: {test_loss:.2f}  ||| Test PA score: {test_pa:2f}    ||| Test mIoU: {test_iou:2f}")

    return inference_data
