from typing import Tuple
def parse_size(size: str) -> int:
    size = size.upper()
    if size.endswith("KB"):
        return int(size[:-2]) * 1024
    elif size.endswith("MB"):
        return int(size[:-2]) * 1024 * 1024
    elif size.endswith("GB"):
        return int(size[:-2]) * 1024 * 1024 * 1024
    elif size.endswith("TB"):
        return int(size[:-2]) * 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Invalid size format: {size}")

def parse_thput(thput: str) -> float:
    thput = thput.upper()
    if thput.endswith("KB/S"):
        return float(thput[:-4]) * 1024
    elif thput.endswith("MB/S"):
        return float(thput[:-4]) * 1024 * 1024
    elif thput.endswith("GB/S"):
        return float(thput[:-4]) * 1024 * 1024 * 1024
    elif thput.endswith("TB/S"):
        return float(thput[:-4]) * 1024 * 1024 * 1024 * 1024
    else:
        raise ValueError(f"Invalid thput format: {thput}")
    

def parse_compression(compression: str) -> Tuple[Tuple[float, float, float, float]]:
    # ratio, quality, encode, decode
    comp_list = [(1.0, 1.0, 0.0, 0.0)]
    if compression.lower() == "none":
        pass
    else:
        # split by comma
        comp_str_list = compression.split(",")
        for comp_str in comp_str_list:
            # split by dash
            comp = comp_str.split("-")
            ratio = float(comp[0])
            quality = float(comp[1])
            encode = float(comp[2])
            decode = float(comp[3])
            comp_list.append((ratio, quality, encode, decode))
    return tuple(comp_list)