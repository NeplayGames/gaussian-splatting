from SaliencyMethods.SaliencyAbstract import SaliencyAbs
from SaliencyMethods.itti import IntensityCenterSurround
from SaliencyMethods.BooleanMap import BooleanMapApprox

SALIENCY_PROCESSORS = {
    "intensitycentersurround": IntensityCenterSurround,
    "itti": IntensityCenterSurround,  # backward-compatible alias; not full Itti-Koch-Niebur
    "booleanmapapprox": BooleanMapApprox,
    "boolean": BooleanMapApprox,  # backward-compatible alias; grayscale approximation
}

def get_saliency_processor(name: str) -> SaliencyAbs:
    cls = SALIENCY_PROCESSORS.get(name.lower())
    if cls is None:
        raise ValueError(f"No such processor: {name}")
    return cls()
