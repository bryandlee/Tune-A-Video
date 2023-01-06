import math
import textwrap
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pad(image: Image.Image, top=0, right=0, bottom=0, left=0, color=(255, 255, 255)):
    new_image = Image.new(image.mode, (image.width + right + left, image.height + top + bottom), color)
    new_image.paste(image, (left, top))
    return new_image


def annotate_text(image: Image.Image, text: str, font_size=15):
    font = ImageFont.truetype("/data/etc/OpenSans-VariableFont_wdth,wght.ttf", size=font_size)

    image_w = image.width
    _, _, text_w, text_h = font.getbbox(text)
    line_size = math.floor(len(text) * image_w / text_w)

    lines = textwrap.wrap(text, width=line_size)
    padding = text_h * len(lines)
    image = pad(image, top=padding + 3)

    ImageDraw.Draw(image).text((0, 0), "\n".join(lines), fill=(0, 0, 0), font=font)
    return image


def make_grid(images, rows=None, cols=None):
    if rows is None:
        assert cols is not None
        rows = math.ceil(len(images) / cols)
    else:
        cols = math.ceil(len(images) / rows)

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        if image.size != (w, h):
            image = image.resize((w, h))
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid
