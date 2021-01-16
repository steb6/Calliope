from compressive_transformer import PositionalEncoding
import numpy as np
from PIL import Image


if __name__ == "__main__":
    comp = PositionalEncoding(32, 0, 50)
    m = comp.pe[0].numpy()
    m = np.repeat(m, 10, axis=0)
    m = np.repeat(m, 10, axis=1)
    mem_formatted = (m * 255 / np.max(m)).astype('uint8')
    mem_img = Image.fromarray(mem_formatted)
    # mem_img.show()
    mem_img.save("pos_enc.png")

    mask = np.tril(np.ones((50, 50)))
    mask = np.repeat(mask, 10, axis=0)
    mask = np.repeat(mask, 10, axis=1)
    mask = np.pad(mask, 1)
    mask_formatted = (mask * 255 / np.max(mask)).astype('uint8')
    mask_img = Image.fromarray(mask_formatted)
    # mem_img.show()
    mask_img.save("mask.png")
