from skimage.draw import polygon
import numpy as np
import plistlib
import matplotlib.pyplot as plt

def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x

    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [load_point(point) for point in points]
            if len(points) <= 2:
                for point in points:
                    mask[int(point[0]), int(point[1])] = 1
            else:
                x, y = zip(*points)
                x, y = np.array(x), np.array(y)
                poly_x, poly_y = polygon(x, y, shape=imshape)
                mask[poly_x, poly_y] = 1
    return mask


# Path to your .roi file
mask_path = "your path"

# Image size of INbreast dataset
imshape = (4084, 3328)

# Load the mask
mask = load_inbreast_mask(mask_path, imshape)

# Verify mask contains non-zero values (indicating ROI regions were properly processed)
print(f"Mask shape: {mask.shape}")
print(f"Non-zero pixel count: {np.count_nonzero(mask)}")

# Display the mask using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(mask, cmap="gray")
plt.title("Loaded ROI Mask")
plt.axis("off")
plt.show()