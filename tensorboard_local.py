import torch
import tensorboardX as tb

class TensorBoard(object):
    def __init__(self, log_dir, log_image_interval):
        self.log_dir = log_dir
        # we don't wan't to log images every iteration,
        # which is expensive, so we can specify `log_image_interval`
        self.log_image_interval = log_image_interval
        self.writer = tb.SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)
        
    def log_image(self, tag, image, step):
        
        # convert to numpy array
        if torch.is_tensor(image):
            image = image.numpy()
        assert image.ndim == 2, "can only log 2d images"
        
        # change the image normalization for the tensorboard logger
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image= image / max_val
    
        self.writer.add_image(tag, img_tensor=image, global_step=step)