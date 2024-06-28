from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def process_image(image_path):
    image = Image.open(image_path)
    pass

image_paths = []

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, image_paths))

#images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]