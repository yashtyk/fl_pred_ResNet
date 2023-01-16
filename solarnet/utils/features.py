
import numpy as np

from scipy.ndimage import center_of_mass

'''
def column_definition(start_pos: int):
    d = dict()
    d['THRESH_POS'] = tb.Float32Col(pos=start_pos+len(d))
    d['THRESH_NEG'] = tb.Float32Col(pos=start_pos+len(d))
    d['NPIX_POS'] = tb.Int32Col(pos=start_pos+len(d))
    d['NPIX_NEG'] = tb.Int32Col(pos=start_pos+len(d))
    d['MAX_POS'] = tb.Float32Col(pos=start_pos+len(d))
    d['MIN_NEG'] = tb.Float32Col(pos=start_pos+len(d))
    d['MEAN_POS'] = tb.Float32Col(pos=start_pos+len(d))
    d['MEAN_NEG'] = tb.Float32Col(pos=start_pos+len(d))
    d['MEDIAN_POS'] = tb.Float32Col(pos=start_pos+len(d))
    d['MEDIAN_NEG'] = tb.Float32Col(pos=start_pos+len(d))
    d['STD_DEV_POS'] = tb.Float32Col(pos=start_pos+len(d))
    d['STD_DEV_NEG'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_POS_X'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_POS_Y'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_DELTA_R'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_DELTA_PHI'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_POS_XR'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_POS_YR'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_NEG_XR'] = tb.Float32Col(pos=start_pos+len(d))
    d['COM_NEG_YR'] = tb.Float32Col(pos=start_pos+len(d))
    d['COMPACTNESS_POS'] = tb.Float32Col(pos=start_pos+len(d))
    d['COMPACTNESS_NEG'] = tb.Float32Col(pos=start_pos+len(d))
    return d

'''
def calculate_features(magnetogram):
    magnetogram1 = magnetogram.detach().squeeze()
    magnetogram = magnetogram1
    #magnetogram: np.ndarray = segments['magnetogram']
    #magnetogram = np.nan_to_num(magnetogram)

    # thresh_pos = filters.threshold_otsu((magnetogram > 0) * magnetogram)
    thresh_pos = 150/256
    # magnetogram_pos_labels = (magnetogram > thresh_pos).astype(int)
    # magnetogram_pos = magnetogram_pos_labels * magnetogram
    magnetogram_pos: np.ma.MaskedArray = np.ma.masked_array(magnetogram, mask=magnetogram < thresh_pos)
    npix_pos = magnetogram_pos.count()
    if npix_pos != 0:
        max_pos = magnetogram_pos.max()
        mean_pos = magnetogram_pos.mean()
        median_pos = np.ma.median(magnetogram_pos)
        std_dev_pos = magnetogram_pos.std()  # nonzero_std(magnetogram_pos)
        com_pos = np.array(center_of_mass(magnetogram_pos))
        com_pos_r = com_pos / np.array(magnetogram.shape)
        compactness_pos = compactness(magnetogram_pos, com_pos, mean_pos)
    else:
        max_pos = thresh_pos
        mean_pos = thresh_pos
        median_pos = thresh_pos
        std_dev_pos = 0
        com_pos = np.array(magnetogram.shape) / 2
        com_pos_r = np.array([0.5, 0.5])
        compactness_pos = 0

    # thresh_neg = filters.threshold_otsu((magnetogram < 0) * magnetogram)
    thresh_neg = 100/256
    # magnetogram_neg_labels = (magnetogram < thresh_neg).astype(int)
    # magnetogram_neg = magnetogram_neg_labels * magnetogram
    magnetogram_neg: np.ma.MaskedArray = np.ma.masked_array(magnetogram, mask=magnetogram > thresh_neg)
    npix_neg = magnetogram_neg.count()
    if npix_neg != 0:
        min_neg = magnetogram_neg.min()
        mean_neg = magnetogram_neg.mean()  # nonzero_mean(magnetogram_neg)
        median_neg = np.ma.median(magnetogram_neg)
        std_dev_neg = magnetogram_neg.std()  # nonzero_std(magnetogram_neg)
        com_neg = np.array(center_of_mass(magnetogram_neg))
        com_neg_r = com_neg / np.array(magnetogram.shape)
        compactness_neg = compactness(magnetogram_neg, com_neg, mean_neg)
    else:
        min_neg = thresh_neg
        mean_neg = thresh_neg
        median_neg = thresh_neg
        std_dev_neg = 0
        com_neg = np.array(magnetogram.shape) / 2
        com_neg_r = np.array([0.5, 0.5])
        compactness_neg = 0

    com_delta = com_neg - com_pos
    com_delta_r = np.sqrt(np.sum(com_delta ** 2))
    com_delta_phi = np.arctan2(com_delta[0], com_delta[1])

    return (
        thresh_pos,
        thresh_neg,
        npix_pos,
        npix_neg,
        max_pos,
        min_neg,
        mean_pos,
        mean_neg,
        median_pos,
        median_neg,
        std_dev_pos,
        std_dev_neg,
        com_pos[0],
        com_pos[1],
        com_delta_r,
        com_delta_phi,
        com_pos_r[0],
        com_pos_r[1],
        com_neg_r[0],
        com_neg_r[1],
        compactness_pos,
        compactness_neg
    )


# def center_of_mass(image: np.ma.MaskedArray):
#     x = range(0, image.shape[1])
#     y = range(0, image.shape[0])
#
#     image_sum = image.sum()
#
#     (X, Y) = np.meshgrid(x, y)
#
#     x_coord = (X*image).sum() / image_sum
#     y_coord = (Y*image).sum() / image_sum
#
#     return np.array([x_coord, y_coord])


def compactness(image: np.ma.MaskedArray, com=None, mean=None):
    com = com if com is not None else center_of_mass(image)
    mean = mean if mean is not None else image.mean()
    nx, ny = image.shape
    x = np.arange(nx) - com[0]
    y = np.arange(ny) - com[1]
    X, Y = np.meshgrid(x, y)
    d_pos = np.sqrt(X ** 2 + Y ** 2).T
    return (d_pos * abs(image)).mean() / abs(mean)
