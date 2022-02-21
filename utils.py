import os
import numpy as np


def txt2ply(txt_file, ply_file=None):
    if ply_file is None:
        ply_file = txt_file.replace('.txt', '.ply')
    with open(txt_file, 'r') as f:
        txt_lines = [line for line in f.readlines() if len(line) > 0]
    ply_lines = [
        'ply','format ascii 1.0',
        'element vertex %d' % len(txt_lines),
        'property float32 x',         
        'property float32 y',         
        'property float32 z',         
        'end_header',
    ]
    for txt_line in txt_lines:
        ply_line = ' '.join(txt_line.split(','))
        ply_lines.append(ply_line.replace('\n', ''))
    os.makedirs(os.path.dirname(ply_file), exist_ok=True)
    with open(ply_file, 'w') as f:
        f.write('\n'.join(ply_lines))

    
def get_angles(U, v):
    if U.ndim == 1:
        U = np.expand_dims(U, 0)
    t = np.linalg.norm(U, axis=1) * np.linalg.norm(v)
    angles = np.abs(np.arccos(np.dot(U, v) / t)) # (N,)
    if angles.shape[0] == 1:
        return angles[0]
    return angles
 
