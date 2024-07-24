import numpy as np
from train_app.loss.metrics import IoU, PixelAccuracy, MAP, MAF1, MAR

class Evaluator(object):
    def __init__(self, num_class):
        kwargs = {'device_fn': lambda x:x,
                  'log_fn':lambda x:x,
                  'log_image_fn':lambda x:x,
                  'log_prefix_fn':lambda x:x}
        self.iou = IoU(K=num_class, **kwargs)
        self.pix = PixelAccuracy(**kwargs)
        self.map = MAP(K=num_class, **kwargs)
        self.mar = MAR(K=num_class, **kwargs)
        self.maf1 = MAF1(K=num_class, **kwargs)

        self.num_class = num_class
        self.scores = {
            "IoU": [],
            "PixelAccuracy": [],
            "MAP": [],
            "MAR": [],
            "MAF1": []

        }
        self.eps = 1e-8

    @property
    def Precision(self):
        return self.scores["MAP"]

    @property
    def Recall(self):
        return self.scores["MAR"]

    @property
    def F1(self):
       return self.scores["MAF1"]

    @property
    def Intersection_over_Union(self):
        return self.scores["IoU"]

    @property
    def pixelAccuracy(self):
        return self.scores["PixelAccuracy"]

    def add_batch(self, gt_image, pre_image):
        # assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
        #                                                                                          gt_image.shape)
        self.scores["IoU"].append(self.iou(pre_image, gt_image).cpu().item())
        self.scores["MAP"].append(self.map(pre_image, gt_image))
        self.scores["MAR"].append(self.mar(pre_image, gt_image))
        self.scores["MAF1"].append(self.maf1(pre_image, gt_image))
        
        self.scores["PixelAccuracy"].append(self.pix(pre_image, gt_image).cpu().item())

    def eval(self):
        self.scores = {k: np.mean(v) for k, v in self.scores.items()}
