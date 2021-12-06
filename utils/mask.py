import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def generate_mask(img_height, img_width, **kwargs):
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

if __name__ == "__main__":
    mask = generate_mask(16, 16)
    print(mask.shape)
    print(mask)
