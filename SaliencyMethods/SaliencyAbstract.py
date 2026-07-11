from abc import ABC, abstractmethod
class SaliencyAbs(ABC):
    @abstractmethod
    def saliency_loss(pred, target):
        pass
    @abstractmethod
    def get_saliency_map(self, image):
        pass    
    @abstractmethod
    def saliency_similarity(self, image, gt_image):
        pass