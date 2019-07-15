import numpy as np
from skimage.measure import label
from multiprocessing.dummy import Pool as ThreadPool

SMOOTH = 0
VERBOSE = False

class Instance(object):
    def __init__(self, imgNp, instID):
        self.instID     = int(instID)
        self.mask = self.getInstanceMask(imgNp, instID)
        self.isMatched = False
        self.matched_instance = None
    
    def getInstanceMask(self, imgNp, instLabel):
        return np.where(imgNp==instLabel,1,0)
    
def iou_numpy(outputs: np.array, labels: np.array):
    assert outputs.shape == labels.shape    
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def calculateAPScore(prediction,target,nThreads=5,IOU_tau=0.5):
    prediction = label(prediction,connectivity=1)
    target = label(target,connectivity=1)
    n_label_target = np.amax(target)
    n_label_prediction  = np.amax(prediction)
    target_inst_list = []
    pred_inst_list = []
    
    for inst in range(1,n_label_prediction+1):
        pred_inst_list.append(Instance(prediction,inst))
    
    for inst in range(1,n_label_target+1):
        target_inst_list.append(Instance(target,inst))
    
    PARALLEL = True
    if nThreads ==0 or nThreads is None:
        PARALLEL = False
        
    #Matching
    if PARALLEL:
        def checkIOUMatch(pInst):
             for tInstance in target_inst_list:
                iou_score = iou_numpy(pInst.mask,tInstance.mask)
                if iou_score>IOU_tau:
                    pInst.isMatched = True
                    pInst.matched_instance = tInstance.instID
                    tInstance.isMatched = True
                    tInstance.matched_instance = pInst.instID
                    break
        
        pool = ThreadPool(nThreads)
        pool.map(checkIOUMatch, pred_inst_list)
        pool.close()
        pool.join()
    else:  
        for pInstance in pred_inst_list:
            for tInstance in target_inst_list:
                iou_score = iou_numpy(pInstance.mask,tInstance.mask)
                if iou_score>IOU_tau:
                    pInstance.isMatched = True
                    pInstance.matched_instance = tInstance.instID
                    tInstance.isMatched = True
                    tInstance.matched_instance = pInstance.instID
                    target_inst_list.remove(tInstance) #to save time
                    break

    #Score calculation
    true_postive = 0
    false_positive = 0
    false_negative = 0
    for pInstance in pred_inst_list:
        if pInstance.isMatched:
            true_postive +=1
        else:
            false_positive +=1
    
    for tInstance in target_inst_list:
        if not tInstance.isMatched:
            false_negative+=1
    if VERBOSE:
        print('FN',false_negative)
        print('TP',true_postive)
        print('FP',false_positive)
    ap_SCORE = true_postive/(true_postive+false_negative+false_positive)
    if VERBOSE:print('AP is',ap_SCORE)
    return ap_SCORE










