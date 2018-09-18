import os
from scipy import misc
import pydicom
import matplotlib.pyplot as plt
import numpy, sys
import numpy as np
#import segment
import skimage


import scipy
import scipy.misc as misc


def load_ct(path):
    import pydicom
    print(path)
    dirs = os.listdir( path )
    n = len(dirs)/700+1

    # slices = [pydicom.read_file(os.path.join(path, filename), force=True) for filename in dirs]
    #Limit the number of input patterns to 700.
    slices = [pydicom.read_file(os.path.join(path, dirs[n*m]), force=True) for m in range(0, len(dirs)/n)]

    # Sort the dicom slices in their respective order
    slices.sort(key=lambda x: int(x.InstanceNumber))
    # Get the pixel values for all the slices
    Slices = np.stack([s.pixel_array for s in slices])
    print(len(Slices))
    
    Slices[Slices == -2000] = 0
    Slices[Slices < 1200] = 0
    Slices[Slices > 4100] = 0
    return Slices


#
def LoadMhd(filename):
    import SimpleITK as sitk
    import numpy as np
    # load image
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    return numpyImage
   
def stl_export(Images,TH,filename, StepSize):
    import numpy as np
    from stl import mesh
    from skimage import measure
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(Images,TH,step_size=StepSize)
    
    # Define the N vertices of the cube
    vertices = verts
    # Define the 12 triangles composing the cube
    faces = faces
    
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]
    
    # Write the mesh to file "cube.stl"
    cube.save(filename)

def load_itk_image(filename): 
     import SimpleITK as sitk 
     itkimage = sitk.ReadImage(filename)
     numpyImage = sitk.GetArrayFromImage(itkimage)
     numpyOrigin = np.array(list(reversed(itkimage.GetOrigin()))) 
     numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
     stretchedVoxelCoord = np.absolute(worldCoord - origin)
     voxelCoord = stretchedVoxelCoord / spacing
     return voxelCoord
 
def normalizePlanes(npzarray):
     maxHU = 400. 
     minHU = -1000.
     npzarray = (npzarray - minHU) / (maxHU - minHU) 
     npzarray[npzarray>1] = 1. 
     npzarray[npzarray<0] = 0. 
     return npzarray
    
def largest_region(binary):
    import numpy
    import skimage.measure
    largest_area = binary*0
    label_image = skimage.measure.label(binary)
    areas = [r.area for r in skimage.measure.regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in skimage.measure.regionprops(label_image):
            if region.area >= areas[-1]:
                for coordinates in region.coords:                
                    largest_area[coordinates[0], coordinates[1], coordinates[2]] = 1
    else:
        largest_area = binary    
                
    return largest_area
