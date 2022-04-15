from functools import reduce
import os
import gdal
import numpy as np
import pandas as pd
import xarray as xr
import osr
from skimage.filters import rank
from skimage.io import imsave
from scipy import ndimage


def destripe(img, dtype=np.uint8, add_val=128): 
    """ 
    Run the de-striping filter proposed by Crippen (Photogramm Eng Remote Sens 55, 1989) on an image. 
 
    :param img: image to de-stripe. 
    :param dtype: original datatype of image 
    :param add_val: constant value to add to keep image within the original bit-depth. 
 
    :returns: filt_img: the filtered image. 
    """ 
    k1 = np.ones((1, 51)) 
    k2 = np.ones((17, 1)) 
    k3 = np.ones((1, 15)) 
 
    F1 = rank.mean(img, selem=k1, mask=img > 0) 
    F2 = F1 - rank.mean(F1, selem=k2, mask=img > 0) + add_val 
    F2[img == 0] = 0 
     
    F3 = rank.mean(F2, selem=k3, mask=img > 0) 
 
    outimg = img.astype(np.float32) - F3 + add_val 
    outimg[outimg > np.iinfo(dtype).max] = np.iinfo(dtype).max 
    outimg[outimg < np.iinfo(dtype).min] = np.iinfo(dtype).min 
    outimg[img == 0] = 0 
 
    return outimg.astype(dtype)


def pca_decomp(img): 
    band_list = [] 
    for b in img: 
        band_list.append(b.reshape(-1, 1)) 
 
    A = np.ma.masked_invalid(np.concatenate(band_list, axis=1)) 
    A -= np.ma.mean(A, 0) 
     
    cov = np.ma.cov(A.T) 
 
    sigma = np.ma.diag(np.sqrt(cov.diagonal())) 
    eigval, V = np.linalg.eig(cov) 
    S = np.ma.diag(1/np.sqrt(eigval)) 
 
    T = reduce(np.dot, [sigma, V, S, V.T]) 
 
    P = np.dot(A, T) 
 
    B = np.zeros(img.shape, dtype=np.float32) 
 
    for i in range(P.shape[1]): 
        p01, p99 = np.percentile(P[np.isfinite(P[:, i]), i], (1, 99)) 
        P[P[:, i] <= p01, i] = p01 
        P[P[:, i] >= p99, i] = p99 
        # B[i, :, :] = (P[:, i].reshape(img.img.shape[1:]) - p01) / (p99 - p01) 
        B[i, :, :] = P[:, i].reshape(img.shape[1:]) - p01 
 
    return B


def parse_object(text):
    out_dict = dict()
    for t in text:
        if 'OBJECT' not in t:
            split = t.split('=')
            out_dict[split[0]] = split[1].replace('"', '')
    return out_dict


def write_image(img, fn_out, dtype, driver='GTiff'):
    driver = gdal.GetDriverByName(driver)
    if img.ndim < 3:
        nrows, ncols = img.shape
        nbands = 1
    else:
        nbands, nrows, ncols = img.shape

    out = driver.Create(fn_out, ncols, nrows, nbands, dtype)

    if nbands == 1:
        out.GetRasterBand(1).WriteArray(img)
    else:
        for b in range(nbands):
            out.GetRasterBand(b+1).WriteArray(img[b])

    out.FlushCache()
    out = None


def get_increments(meta):
    raw = [l.strip() for l in meta.split('\n')]

    map_start = raw.index('GROUP=DimensionMap')
    map_end = raw.index('END_GROUP=DimensionMap')

    dim_map = raw[map_start:map_end+1]
    dim1 = parse_object(dim_map[dim_map.index('OBJECT=DimensionMap_1'):dim_map.index('END_OBJECT=DimensionMap_1')+1])
    dim2 = parse_object(dim_map[dim_map.index('OBJECT=DimensionMap_2'):dim_map.index('END_OBJECT=DimensionMap_2')+1])

    if dim1['GeoDimension'] == 'GeoTrack':
        row_inc = int(dim1['Increment'])
        col_inc = int(dim2['Increment'])
    else:
        row_inc = int(dim2['Increment'])
        col_inc = int(dim1['Increment'])

    return row_inc, col_inc


def filter_bands(bands):
    filtered = []
    for band in bands:
        pad = ndimage.maximum_filter(band, size=20)
        b = band.copy()
        b[band == 0] = pad[band == 0]
        nostripe = destripe(b, dtype=np.uint16, add_val=int(b[b > 0].mean()))

        filtered.append(ndimage.gaussian_filter(nostripe, 1, mode='nearest'))
    return np.array(filtered).astype(np.float64)


def process_pca(filename, outfilename, epsg, pixelSpacing):
    ds = xr.open_dataset(filename)
    bands = np.array(ds[['Band13', 'Band12', 'Band10']].to_array())

    filtered = filter_bands(bands)
    filtered[bands == 0] = np.nan

    pca = pca_decomp(filtered).astype(np.int16)
    pca[np.isnan(filtered)] = -9999

    georeference_img(pca, ds, ndvalue=-9999)

    warp_image(outfilename, epsg, pixelSpacing)


def filter_emissivity(filename, outfilename, epsg, pixelSpacing):
    ds = xr.open_dataset(filename)
    bands = np.array(ds[['Band14', 'Band13', 'Band12', 'Band11', 'Band10']].to_array())

    filtered = filter_bands(bands)
    filtered[bands == 0] = 0

    georeference_img(filtered, ds, gdal.GDT_Int16, ndvalue=0)

    warp_image(outfilename, epsg, pixelSpacing)


def georeference_img(img, ds, dtype, ndvalue=None):
    write_image(img, 'tmp.tif', dtype)

    row_inc, col_inc = get_increments(ds.attrs['StructMetadata.0'])
    if img.ndim < 3:
        rows = np.arange(0, img.shape[0] + 1, row_inc)
        cols = np.arange(0, img.shape[1] + 1, col_inc)
        nbands = 1
    else:
        rows = np.arange(0, img[0].shape[0] + 1, row_inc)
        cols = np.arange(0, img[0].shape[1] + 1, col_inc)
        nbands = img.shape[0]

    C, R = np.meshgrid(cols, rows)

    longitudes = ds['Longitude'].values
    latitudes = ds['GeodeticLatitude'].values

    gcp_df = pd.DataFrame()
    gcp_df['pixel'] = C.flatten()
    gcp_df['line'] = R.flatten()
    gcp_df['x'] = longitudes.flatten()
    gcp_df['y'] = latitudes.flatten()

    gcp_list = []
    for i, row in gcp_df.iterrows():
        gcp = gdal.GCP(row.x, row.y, 0, row.pixel, row.line)
        gcp_list.append(gcp)

    crs = osr.SpatialReference()
    crs.ImportFromEPSG(4326)

    out_ds = gdal.Open('tmp.tif', gdal.GA_Update)
    out_ds.SetGCPs(gcp_list, crs.ExportToWkt())

    if ndvalue is not None:
        for b in range(nbands):
            out_ds.GetRasterBand(b+1).SetNoDataValue(ndvalue)

    out_ds.FlushCache()
    out_ds = None


def warp_image(outfilename, epsg, pixelSpacing):
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(4326)

    dst_crs = osr.SpatialReference()
    dst_crs.ImportFromEPSG(epsg)

    gdal.Warp(outfilename, 'tmp.tif', srcSRS=crs, dstSRS=dst_crs, xRes=pixelSpacing, yRes=pixelSpacing)

    os.remove('tmp.tif')
