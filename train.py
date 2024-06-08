import os 
import torch
import tqdm
from metrics import Metrics

def train_validation(model, tr_dl, val_dl, n_cls, loss_fn, optim, epochs, device, save_dir, save_prefix):
    
    tr_loss_list, tr_pa_list, tr_iou_list = [],[],[] 
    val_loss_list, val_pa_list, val_iou_list = [],[],[] 
    tr_len, val_len = len(tr_dl), len(val_dl)
    
    best_loss = 0
    os.makedirs(f"{save_dir}", exist_ok=True)
    for epoch in range(epochs): 
        print("-------------------------------------------------")
        print(f"{epoch+1} - EPOCH PROCESS STARTED....".upper())   
        model.train()

        tr_epoch_loss, tr_epoch_pa, tr_epoch_iou = 0, 0, 0   
        print(f"TRAINING...".upper())  
        for (images, masks) in tqdm.tqdm(tr_dl): 
             
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            
            #get metrics in each batch per loop
            met = Metrics(pred=predictions, gt=masks, loss_fn=loss_fn, n_cls=n_cls)
            losses = met.loss()
            #har bir batchlarni metricslarini qoshib ketad. 10 ta batch bolsa 10 tasini summasini chiqarida
            tr_epoch_loss += losses.item()
            tr_epoch_pa += met.PA()
            tr_epoch_iou += met.mIoU() 

            optim.zero_grad()
            losses.backward()
            optim.step()


        """VALDIATION """
        print(f"VALIDATIONING...".upper())
        model.eval()
        val_epoch_loss, val_epoch_pa, val_epoch_iou = 0, 0, 0
        with torch.no_grad():       
            for (images, masks) in tqdm.tqdm(val_dl): 
                images, masks = images.to(device), masks.to(device)
                predictions = model(images)
                
                #get metrics in each batch 
                met = Metrics(pred=predictions, gt=masks, loss_fn=loss_fn)
                losses = met.loss()

                #har bir val batchlarni metricslarni qoshib ketadi
                val_epoch_loss += losses.item()
                val_epoch_pa += met.PA()
                val_epoch_iou += met.mIoU() 
        

        #Har bir epoch uchun loss, pixel acc, va iou ni topib olamiz. 
        #Buning uchun tr loss, pa, mio ni jami data soniga bo'lamiz. 
        tr_epoch_loss /= tr_len
        tr_epoch_pa /= tr_len
        tr_epoch_iou /= tr_len

        val_epoch_loss /= val_len
        val_epoch_pa /= val_len
        val_epoch_iou /= val_len
        
        #Har bir epochdagi train, val loss va accuracy natijalarni listga store qilib ketamiz. 
        tr_loss_list.append(tr_epoch_loss) 
        tr_pa_list.append(tr_epoch_pa)
        tr_iou_list.append(tr_epoch_iou)

        val_loss_list.append(val_epoch_loss) 
        val_pa_list.append(val_epoch_pa)
        val_iou_list.append(val_epoch_iou)

            
        print("-------------------------------------------------")
        print(f"TRAINING RESULTS:")
        print(f"Train loss : {tr_epoch_loss:.3f}")
        print(f"Train PA   : {tr_epoch_pa:.3f}")
        print(f"Train mIoU : {tr_epoch_iou:.3f}\n")

        print(f"VALDIATION RESULTS:")
        print(f"Validation loss : {val_epoch_loss:.3f}")
        print(f"Validation PA   : {val_epoch_pa:.3f}")
        print(f"Validation mIoU : {val_epoch_iou:.3f}")

        print(f"{epoch+1} - EPOCH PROCESS FINISHED....".upper())   
        print("-------------------------------------------------")

        if val_epoch_loss > best_loss: 
            best_loss = val_epoch_loss 
            best_model = model.state_dict() #whole status of model in dict
            torch.save(best_model, f=f"{save_dir}/{save_prefix}_best_model.pth")
        
    return {"tr_loss": tr_loss_list, "tr_pa": tr_pa_list, "tr_iou": tr_iou_list,
            "val_loss": val_loss_list, "val_pa": val_pa_list, "val_iou": val_iou_list}

