import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
from skimage.morphology import skeletonize
import plotly.graph_objects as go
from scipy.spatial import distance
import os
import argparse
import trimesh


# --- 1. Extract strokes from 2D sketches ---
def preprocess_sketch(img_array):
    # Convert sketch to binary using a threshold
    _, binary_img = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_img

def detect_strokes(binary_img):
    # Skeletonize the binary image
    skeleton = skeletonize(binary_img / 255) * 255

    # Flip the skeletonized image vertically
    #skeleton = np.flipud(skeleton)

    # Find contours in the skeletonized image
    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def parameterize_strokes(contours):
    # Store strokes and their parameters
    strokes = []

    for contour in contours:
        # Filter out very small contours
        if len(contour) > 5:
            strokes.append({
                "control_points": contour,
                # Further attributes can be added as needed
            })

    return strokes

def post_process(strokes):
    # Refine the extracted strokes. For this example, we'll just return the strokes as they are.
    # More advanced post-processing (e.g., smoothing) can be implemented as needed.
    return strokes

def project_sketch_on_cube(img_path, plane='xz'):
    """Load, preprocess, and project a sketch onto a face of a 3D cube."""
    img = plt.imread(img_path).astype(float) / 255.0

    # Split the image into the input and sketch sections
    height = img.shape[0]
    start_y = int(height * 0.6)  # Start from 60% of the height to cut off the top 20% of the sketch
    bottom_half = img[start_y:, :]

    # Cut the bottom half
    #half_index = img.shape[0] // 2
    #bottom_half = img[half_index:]

    # Resize the bottom half to match the original dimensions
    zoom_factor = [2, 1, 1]
    resized_bottom_half = zoom(bottom_half, zoom_factor)

    # Clip the pixel values to be within [0, 1]
    resized_bottom_half = np.clip(resized_bottom_half, 0, 1)

    scale_factor = 300

    if plane == 'xz':
        # Only flip vertically
        resized_bottom_half = np.flipud(resized_bottom_half)

        x, z = np.meshgrid(np.linspace(-1, 1, resized_bottom_half.shape[1]),
                           np.linspace(-1, 1, resized_bottom_half.shape[0]))
        y = -np.ones_like(x)  # Moved to the opposite side

        return x, y, z, resized_bottom_half

    elif plane == 'yz':
        # Flip the image vertically for correct orientation
        resized_bottom_half = np.flipud(resized_bottom_half)

        x = np.ones_like(resized_bottom_half[:,:,0])  # On the right face
        y, z = np.meshgrid(np.linspace(-1, 1, resized_bottom_half.shape[1]),
                           np.linspace(-1, 1, resized_bottom_half.shape[0]))

        return x, y, z, resized_bottom_half

    elif plane == 'hypotenuse':
        # Flip both vertically and horizontally
        resized_bottom_half = np.flipud(np.fliplr(resized_bottom_half))

        x_vals = np.linspace(1, -1, resized_bottom_half.shape[1])
        y_vals = x_vals  # Diagonal from bottom-right to top-left
        z_vals = np.linspace(-1, 1, resized_bottom_half.shape[0])
        x, z = np.meshgrid(x_vals, z_vals)
        y = np.ones_like(x) * y_vals

        return x, y, z, resized_bottom_half

# --- 2. Modify strokes ---
def plot_and_return_modified_strokes(ax, strokes, plane='xz'):
    modified_strokes = []
    for stroke in strokes:
        control_points = stroke["control_points"]
        modified_stroke = {}
        if plane == 'xz':
            xs = control_points[:, 0, 0] - 150
            ys = control_points[:, 0, 1]
            # Map to the xz plane
            ax.plot(xs, [-1] * len(xs), ys, color='blue')
            modified_stroke["xs"] = xs
            modified_stroke["ys"] = [-1] * len(xs)
            modified_stroke["zs"] = ys
        elif plane == 'yz':
            xs = control_points[:, 0, 0] - 150
            ys = control_points[:, 0, 1]
            # Map to the yz plane
            ax.plot([1] * len(xs), xs, ys, color='red')
            modified_stroke["xs"] = [1] * len(xs)
            modified_stroke["ys"] = xs
            modified_stroke["zs"] = ys
        modified_strokes.append(modified_stroke)
    return modified_strokes

# --- 3. Plot modified strokes with projection ---
def plot_modified_strokes_with_projections_on_3d(ax, modified_strokes, plane='xz'):
    projected_strokes = []

    for stroke in modified_strokes:
        # Plot the modified stroke
        ax.plot(stroke["xs"], stroke["ys"], stroke["zs"], color='blue' if plane == 'xz' else 'red')

        # Draw projection lines
        if plane == 'xz':
            for x, y, z in zip(stroke["xs"], stroke["ys"], stroke["zs"]):
                ax.plot([x, x], [300, y], [z, z], color='blue', linestyle='-')
                projected_strokes.append({'start': (x, y, z), 'end': (x, 300, z)})

        elif plane == 'yz':
            for x, y, z in zip(stroke["xs"], stroke["ys"], stroke["zs"]):
                ax.plot([300, x], [y, y], [z, z], color='red', linestyle='-')
                projected_strokes.append({'start': (x, y, z), 'end': (300, y, z)})

    return projected_strokes

# --- 4. find intersections of projection  ---
def lines_intersect_3D(P1, P2, Q1, Q2):
    # Calculate direction vectors
    d1 = np.array(P2) - np.array(P1)
    d2 = np.array(Q2) - np.array(Q1)

    # Cross product of direction vectors
    cross_d = np.cross(d1, d2)

    # If cross product is zero, lines are parallel or collinear
    if np.linalg.norm(cross_d) == 0:
        return None

    # Check coplanarity
    if np.dot(cross_d, (np.array(Q1) - np.array(P1))) != 0:
        return None

    # Calculate the 3x3 matrix and its determinant
    T = np.array(Q1) - np.array(P1)
    M = np.column_stack((d1, -d2, cross_d))
    det_M = np.linalg.det(M)

    # If determinant is zero, lines are not coplanar
    if det_M == 0:
        return None

    # Calculate the parameters s and t for the line equations
    s = np.linalg.det(np.column_stack((T, -d2, cross_d))) / det_M
    t = np.linalg.det(np.column_stack((d1, T, cross_d))) / det_M

    # If 0 <= s, t <= 1, then there's an intersection within the line segments
    if 0 <= s <= 1 and 0 <= t <= 1:
        intersection_point = P1 + s * d1
        return tuple(intersection_point)
    return None

def find_intersections(strokes):
    """
    Finds all intersection points among the provided strokes.

    Args:
    - strokes (list): List of strokes.

    Returns:
    - List of intersection points.
    """
    intersections = []
    for i in range(len(strokes)):
        for j in range(i+1, len(strokes)):
            intersection = lines_intersect_3D(strokes[i]['start'], strokes[i]['end'],
                                              strokes[j]['start'], strokes[j]['end'])
            if intersection:
                intersections.append(intersection)
    return intersections

# --- 5. Connect Intersections and Visualize ---
def connect_intersections(intersections):
    lines = []
    remaining_points = intersections.copy()
    current_point = remaining_points.pop(0)
    while remaining_points:
        distances = [np.linalg.norm(np.array(p) - np.array(current_point)) for p in remaining_points]
        nearest_point_idx = distances.index(min(distances))
        nearest_point = remaining_points.pop(nearest_point_idx)
        lines.append((current_point, nearest_point))
        current_point = nearest_point
    return lines

def plot_3d_sketch(intersections):
    lines = connect_intersections(intersections)
    fig = go.Figure()
    for line in lines:
        start, end = line
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines+markers',
                line=dict(color='red', width=3)
            )
        )
    fig.update_layout(scene=dict(aspectmode="cube"))
    fig.show()

def save_as_obj(intersections, lines, filename):
    with open(filename, 'w') as obj_file:
        for point in intersections:
            obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")
        for line in lines:
            start_idx = intersections.index(line[0]) + 1
            end_idx = intersections.index(line[1]) + 1
            obj_file.write(f"l {start_idx} {end_idx}\n")

def save_points_as_obj(points, filename):
    """
    Saves a list of points as a .obj file.

    Args:
    - points (list): List of points.
    - filename (str): Name of the output .obj file.
    """
    with open(filename, 'w') as obj_file:
        # Write vertices to the .obj file
        for point in points:
            obj_file.write("v {} {} {}\n".format(point[0], point[1], point[2]))



if __name__ == "__main__":

    """  parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path of 3d model')

    args, unknown = parser.parse_known_args()  # ignore unrecognized arguments """

    abs_path = os.path.abspath(os.getcwd())

    # Sample execution can be placed here.
    # For example:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Front sketch
    x, y, z, color_front = project_sketch_on_cube(f"{abs_path}/Final_sketches/front_sketch.jpg", 'xz')
    ax.plot_surface(x, y, z, facecolors=color_front, rstride=1, cstride=1, shade=False)

    # Side sketch
    x, y, z, color_side = project_sketch_on_cube(f"{abs_path}/Final_sketches/side_sketch.jpg", 'yz')
    ax.plot_surface(x, y, z, facecolors=color_side, rstride=1, cstride=1, shade=False)

    # For the front sketch:
    front_binary_img = preprocess_sketch((color_front[:,:,0]*255).astype(np.uint8))
    contours_front = detect_strokes(front_binary_img)
    strokes_front = parameterize_strokes(contours_front)
    front_refined_strokes = post_process(strokes_front)

    # For the side sketch:
    side_binary_img = preprocess_sketch((color_side[:,:,0]*255).astype(np.uint8))
    contours_side = detect_strokes(side_binary_img)
    strokes_side = parameterize_strokes(contours_side)
    side_refined_strokes = post_process(strokes_side)

    # (code for 3D visualization)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')

    ax1.set_xlim([0, 300])
    ax1.set_ylim([0, 350])
    ax1.set_zlim([50, 400])

    # Plot the refined strokes and get the modified strokes
    modified_front_strokes = plot_and_return_modified_strokes(ax1, front_refined_strokes, 'xz')
    modified_side_strokes = plot_and_return_modified_strokes(ax1, side_refined_strokes, 'yz')

    # Assuming modified_front_strokes and modified_side_strokes are already available from previous code
    projected_front_strokes = plot_modified_strokes_with_projections_on_3d(ax1, modified_front_strokes, 'xz')
    projected_side_strokes = plot_modified_strokes_with_projections_on_3d(ax1, modified_side_strokes, 'yz')

    # Merge the two stroke lists
    all_strokes = projected_front_strokes + projected_side_strokes

    # Find intersections for the merged strokes
    intersections = find_intersections(all_strokes)

    plot_3d_sketch(intersections)
    lines = connect_intersections(intersections)
    save_as_obj(intersections, lines, f"{abs_path}/Final_sketches/3d_edge_sketch.obj")

    # Sample usage:
    save_points_as_obj(intersections, f"{abs_path}/Final_sketches/3d_points_sketch.obj")
