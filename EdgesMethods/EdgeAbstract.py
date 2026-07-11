from abc import ABC, abstractmethod
class EdgeAbs(ABC):
    @abstractmethod
    def edge_loss(pred, target):
        pass
    @abstractmethod
    def get_edge_map(self, target):
        pass    
    @abstractmethod
    def edge_similarity(pred, target):
        pass