
from EdgesMethods.EdgeAbstract import EdgeAbs
from EdgesMethods.Sobel import Sobel
EDGE_PROCESSORS = {
    "sobel": Sobel,
}

def get_edge_processor(name: str) -> EdgeAbs:
    cls = EDGE_PROCESSORS.get(name.lower())
    if cls is None:
        raise ValueError(f"No such processor: {name}")
    return cls()
