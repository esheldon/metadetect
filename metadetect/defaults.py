BMASK_EDGE = 2**30
DEFAULT_IMAGE_VALUES = {
    'image': 0.0,
    'weight': 0.0,
    'seg': 0,
    'bmask': BMASK_EDGE,
    'noise': 0.0,
}

ALLOWED_BOX_SIZES = [
    2, 4, 6, 8, 12, 16, 24, 32, 48,
    64, 96, 128, 192, 256,
    384, 512, 768, 1024, 1536,
    2048, 3072, 4096, 6144
]
