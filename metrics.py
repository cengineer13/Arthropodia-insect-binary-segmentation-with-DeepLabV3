import torch
import numpy as np

class Metrics():
    def __init__ (self, pred , gt, loss_fn , n_cls = 2):
 
        self.pred=torch.argmax(torch.nn.functional.softmax (pred, dim = 1), dim = 1 )
        self.gt = gt 
        self.loss_fn = loss_fn 
        self.n_cls = n_cls 
        self.pred_ = pred 
        
    def loss(self) :   return  self.loss_fn(self.pred_, self.gt)       
    
    def to_contiguous (self,inp) : 
        """ Runtime: input is not contigious erroridan qochish uchun. Yani ayrim view, transpose kabi  funksiyalarni ishlatganda
         xotira adressini ozgartiradi. contiguous ni ishlatsak xotirani ozgartirmasdan yangi tensor kopiya qiladi  """
        return inp.contiguous().view(-1)
    
    def PA (self):
        """Pixel accuracy. Pixel accuracy = (To'gri topilgan pixellar sonini) / (Rasmdagi barcha pixellar soniga)"""
        with torch.no_grad():
            match = torch.eq(self.pred , self.gt).int()  #pred va gt ni bir biriga togrilarini topish
        return float(match.sum()) / float(match.numel()) 
    
    def mIoU (self):
        """mean Intersection over Union"""
        
        with torch.no_grad():
            pred , gt = self.to_contiguous (self.pred), self.to_contiguous(self.gt)
            iou_per_class = []
            
            for c in range (self.n_cls):
                #prediction va gt ni n_clas dagilarga mosligini tekshirib koradi 
                #masalan: n_cls 2 bolsa yani 0 va 1. har birini 0 yoki 1 ga tengligini tekshirib chiqadi
                match_pred = pred == c 
                match_gt = gt == c 
                
                #gt larni mos kelish yigindisi 0 ga teng bolib qolsa, yanikim 2 ta pixel valueadan boshqa paydo bolib qolsa
                #boshqa rang yoki boshqa narsa
                if match_gt.long().sum().item() == 0 : iou_per_class.append(np.nan) #np.nan - not a number nan qiymat qaytaradi
                else : 
                        
                    #logical_and - kesishgan qismini olish. Ikkovida ham mavjud qism
                    #logical_or - ikkovida ham mavjud bolsa bolmasa olinadigon qism
                    intersection = torch.logical_and (match_pred , match_gt).sum().float().item() 
                    union = torch.logical_or (match_pred , match_gt).sum().float().item()
                    
                    iou = (intersection ) / (union)
                    iou_per_class.append(iou) #har bir class uchun alohida yoziladi. 1-0 klass uchun 2-loopda 1 klass uchun
                    
            return np.nanmean(iou_per_class)