
from SaliencyMethods.SaliencyAbstract import SaliencyAbs
from SaliencyMethods.itti import Itti
from SaliencyMethods.BooleanMap import BooleanMap
SALIENCY_PROCESSORS = {
    "itti": Itti,
    "boolean" : BooleanMap
}

def get_saliency_processor(name: str) -> SaliencyAbs:
    cls = SALIENCY_PROCESSORS.get(name.lower())
    if cls is None:
        raise ValueError(f"No such processor: {name}")
    return cls()
