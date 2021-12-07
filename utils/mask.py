import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import math, random
from PIL import Image, ImageDraw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def generate_spiky_mask(img_height, img_width, **kwargs):
    '''
    create a random polygon mask where mask = 1 is the region to be ignored/masked

    Polygon will be centered around (ctrX, ctrY) i.e image center and with average radius as img_height//4
    
    '''
    height_mid, width_mid = img_height//2, img_width//2
    verts = generatePolygon( ctrX=width_mid, ctrY=height_mid, aveRadius=img_height//4, irregularity=0.35, spikeyness=0.2, numVerts=16 )
    polygon = Polygon(verts)


    mask = np.zeros((img_width,img_height))
    #find the extremes of the polygon and only check for inside points using .contains() to reduce time complexity
    x_coord = [pt[0] for pt in verts]
    y_coord = [pt[1] for pt in verts]
    x_min, x_max = min(x_coord), max(x_coord)
    y_min, y_max = min(y_coord), max(y_coord)

    h_mesh, w_mesh = np.meshgrid(range(y_min, y_max), range(x_min, x_max), indexing='ij')
    h_grid, w_grid = h_mesh.ravel(), w_mesh.ravel()


    for i in range(len(h_grid)):
        point = Point(h_grid[i], w_grid[i])
        if polygon.contains(point):
            mask[h_grid[i], w_grid[i]] = 1
    return mask


# https://newbedev.com/algorithm-to-generate-random-2d-polygon 
def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ) :
    '''Start with the centre of the polygon at ctrX, ctrY, 
    then creates the polygon by sampling points on a circle around the centre. 
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    spikeyness = clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*math.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2*math.pi)
    for i in range(numVerts) :
        r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*math.cos(angle)
        y = ctrY + r_i*math.sin(angle)
        points.append( (int(x),int(y)) )

        angle = angle + angleSteps[i]

    return points

def clip(x, min, max) :
    if( min > max ) :  return x    
    elif( x < min ) :  return min
    elif( x > max ) :  return max
    else :             return x


def generate_convex_hull_mask(img_height, img_width, **kwargs):
    '''create a random polygon mask where mask = 1 is the region to be ignored/masked'''
    low_h, high_h = img_height//3, 2*img_height//3
    low_w, high_w = img_width//3, 2*img_width//3
    rng = np.random.default_rng()
    points = rng.integers([low_h,low_w], [high_h,high_w], size=(20,2))
    hull = ConvexHull(points)
    
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    
    #coordinates = list(np.ndindex(img_width,img_height))
    mask = np.zeros((img_width,img_height))
    
    h_mesh, w_mesh = np.meshgrid(range(low_h, high_h), range(low_w, high_w), indexing='ij')
    h_grid, w_grid = h_mesh.ravel(), w_mesh.ravel()
    #coordinate_grid = np.vstack([h_mesh.ravel(), w_mesh.ravel()])

    for i in range(len(h_grid)):
        if point_in_hull((h_grid[i], w_grid[i]), hull):
            mask[h_grid[i], w_grid[i]] = 1
    return mask


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)



if __name__ == "__main__":
    ## Convex hull mask
    # mask = generate_mask(16, 16)
    # print(mask.shape)
    # print(mask)

    ## Spiky mask
    mask_spiky = generate_spiky_mask(1024,1024)
    print(mask_spiky.shape)
    print(mask_spiky)
