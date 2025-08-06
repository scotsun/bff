import os
import numpy as np

def parse_tags_from_filename(filename: str):
    """
    Parse a filename (without the .pth extension) into a dictionary of tags.
    This function can be adapted to handle as many tags as you need:
      - modality_checkpoint
      - mixing_approach
      - contrast (boolean)
      - zp_only (boolean)
    """
    # Remove .json and split on underscore
    base_name = filename.replace('.pth', '')
    parts = base_name.split('_')
    # print(parts)
    # Initialize a dictionary to store our extracted tags
    tag_dict = {
        'filename': filename,             # store original filename
        'modality_checkpoint': None,
        'mixing_approach': None,
        'contrast': False,
        'zp_only': False
    }
    
    # Go through each part of the file name
    for t in ["first_check", "mid_check", "final_check"]:
        if t in filename:
            tag_dict["modality_checkpoint"] = t
    for p in parts:
        # Check for known tags
        if p in ['softmax-gating', 'masked-avg', 'self-attention']:
            tag_dict['mixing_approach'] = p
        elif p == 'contrast':
            tag_dict['contrast'] = True
        elif p == 'zpOnly':
            tag_dict['zp_only'] = True
    
    return tag_dict


def select_model(root: str, filtering_tags: dict, file_extension: str = "pth", n: None | int = 5):
    """
    filtering_tags = {
        "modality_checkpoint": modality_checkpoint (str),
        "mixing_approach": mixing_approach (str),
        "contrast": True/False,
        "zp_only": True/False
    }
    """
    files = [f for f in os.listdir(root) if f.endswith(file_extension)]
    selected = []
    for f in files:
        tags = parse_tags_from_filename(f)
        if (tags['modality_checkpoint'] == filtering_tags['modality_checkpoint'] 
            and tags['mixing_approach'] == filtering_tags['mixing_approach']
            and tags['contrast'] == filtering_tags['contrast']
            and tags['zp_only'] == filtering_tags['zp_only']):
            selected.append(f)
    if len(selected) < n:
        raise ValueError("model samples not enough")
    if n:
        return np.random.choice(selected, size=n, replace=False).tolist()
    return selected