import os
from PIL import Image

def concat(im1, im2, resize):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    # resize
    dst = dst.resize(
        (int(resize*im1.width), int(resize*dst.height)), Image.ANTIALIAS)
    return dst

def main():
    DIRECTORY = 'results'
    path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    files = os.listdir(path)
    for file in files:
        if file.split('.')[-1] == 'png' and 'tfci' not in file:
            reconstructed = file + '.tfci.png'
            im1 = Image.open(file)
            im2 = Image.open(reconstructed)
            res = concat(im1, im2, 0.5)
            res.save(path + '/' + DIRECTORY + '/' + file)
    

if __name__ == "__main__":
    main()
