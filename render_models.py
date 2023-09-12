import bpy
import os
import json
import argparse
import math
import random
import subprocess

def main(args):
    # Later in the script, you can use args.model_name to get the model name
    # For example:
    print(f'Model name: {args.model_name}')

    abs_path = os.path.abspath(os.getcwd())

    # Define the path to the root directory containing the taxonomy file and model folders
    root_dir = f"{abs_path}/Dataset/ShapeNetCore.v2/ShapeNetCore.v2"

    # Define the path to the taxonomy file
    taxonomy_file_path = os.path.join(root_dir, "taxonomy.json")

    # Open and load the taxonomy file
    with open(taxonomy_file_path, 'r') as f:
        taxonomy = json.load(f)

    # Iterate over the taxonomy objects
    for taxon in taxonomy:
        if args.model_name.lower() in taxon['name'].lower():
            synsetId = taxon['synsetId']
            # Define the path to the model directory for this object
            model_dir = os.path.join(root_dir, synsetId)

            # Check if the model directory exists
            if os.path.exists(model_dir):
                # Get a list of all directories within the model directory
                all_model_ids = os.listdir(model_dir)
                
                # Select a random directory from the list
                model_id = random.choice(all_model_ids)
    
                # Define the path to the .obj file for this model
                model_mtl_file_path = os.path.join(model_dir, model_id, "models/model_normalized.mtl")
                # model_mtl_file_path = os.path.join("/home/sanket/Dataset/ShapeNetCore.v2/ShapeNetCore.v2/03001627/4a0f1aa6a24c889dc2f927df125f5ce4/", "models/model_normalized.mtl")
                # Check if the .obj file exists
                if os.path.exists(model_mtl_file_path):
                    with open(model_mtl_file_path, 'r') as f:
                        lines = f.readlines()
                
                    with open(model_mtl_file_path, 'w') as f:
                        for line in lines:
                            if line.startswith('map_Kd'):
                                f.write('#' + line)
                            else:
                                f.write(line)

                # Define the path to the .obj file for this model
                model_obj_file_path = os.path.join(model_dir, model_id, "models/model_normalized.obj")
                # model_obj_file_path = os.path.join("/home/sanket/Dataset/ShapeNetCore.v2/ShapeNetCore.v2/03001627/4a0f1aa6a24c889dc2f927df125f5ce4/", "models/model_normalized.obj")
                # Check if the .obj file exists
                if os.path.exists(model_obj_file_path):
                        
                    # Your blender operations go here
                    bpy.ops.scene.new(type="NEW")
                    scene = bpy.context.scene  # Get the current scene

                    # Create a new world and set its color to white
                    world = bpy.data.worlds.new("World")
                    # world.color = (1, 1, 1)  # RGB for white
                    scene.world = world

                    # Remove the default cube
                    bpy.ops.object.select_all(action='DESELECT')
                    bpy.ops.object.select_by_type(type='MESH')
                    bpy.ops.object.delete()

                    bpy.ops.import_scene.obj(filepath=model_obj_file_path)
                    obj = bpy.context.selected_objects[0]  # Select the imported object
                    obj.rotation_euler = (math.radians(270), math.radians(180), 0)

                    # Center and scale the model
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
                    bpy.ops.object.location_clear()

                    bpy.context.scene.view_settings.view_transform = 'Standard'

                    # Add camera
                    bpy.ops.object.camera_add(location=[0, 5, 0])
                    camera = bpy.context.object  # Get the camera object

                    # Make sure the newly added camera is the active one
                    bpy.context.scene.camera = camera

                    # Set camera to orthographic mode
                    camera.data.type = 'ORTHO'
                    camera.data.ortho_scale = 1.7

                    bpy.context.scene.render.resolution_x = 1920
                    bpy.context.scene.render.resolution_y = 1080

                    # Create the dictionary to hold the view settings
                    views = {
                        'front': {'camera_location': (0, -2, 0), 'camera_rotation': (math.radians(90), 0, 0), 'light_location': (0, -2, 0)},
                        '30degree': {'camera_location': (2.1, -3.5, 0), 'camera_rotation': (math.radians(90), 0, math.radians(30)), 'light_location': (2.1, -3.5, 0)},
                        'side': {'camera_location': (2, 0, 0), 'camera_rotation': (math.radians(90), 0, math.radians(90)), 'light_location': (2, 0, 0)}
                    }

                    # Iterate over the views
                    for view_name, view_settings in views.items():
                        # Position the camera
                        camera.location = view_settings['camera_location']
                        camera.rotation_euler = view_settings['camera_rotation']
                        
                        # Delete any existing light sources
                        bpy.ops.object.select_by_type(type='LIGHT')
                        bpy.ops.object.delete()
                        
                        # Add a new light source at the specified location
                        bpy.ops.object.light_add(type='SUN', location=view_settings['light_location'])
                        light = bpy.context.object  # Get the light source object
                        light.data.energy = 10  # Increase energy
                        
                        # Render the scene and save the image
                        output_image_name = "{}_{}.png".format(args.model_name, view_name)
                        output_image_path = "/home/sanket/Project/CLIPasso/target_images/" + output_image_name
                        bpy.context.scene.render.filepath = output_image_path
                        bpy.ops.render.render(write_still=True)

                        # Call the run_object_sketching.py script
                        cmd = [
                            'python', 'run_object_sketching.py', 
                            '--target_file', output_image_name, 
                            '--num_strokes', str(args.num_strokes), 
                            '--num_sketches', str(args.num_sketches),
                        ]
                        subprocess.run(cmd, check=True)

                        cmd = [
                            'python', '2d_sketch_to_3d_sketch.py'
                        ]
                        subprocess.run(cmd, check=True)

    

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='The name of the model to render')
    parser.add_argument("--num_strokes", type=int, default=16,
                        help="number of strokes used to generate the sketch, this defines the level of abstraction.")
    parser.add_argument("--num_iter", type=int, default=2001,
                        help="number of iterations")
    parser.add_argument("--fix_scale", type=int, default=0,
                        help="if the target image is not squared, it is recommended to fix the scale")
    parser.add_argument("--mask_object", type=int, default=0,
                        help="if the target image contains background, it's better to mask it out")
    parser.add_argument("--num_sketches", type=int, default=3,
                        help="it is recommended to draw 3 sketches and automatically chose the best one")
    parser.add_argument("--multiprocess", type=int, default=0,
                        help="recommended to use multiprocess if your computer has enough memory")
    parser.add_argument('-colab', action='store_true')
    parser.add_argument('-cpu', action='store_true')
    parser.add_argument('-display', action='store_true')
    parser.add_argument('--gpunum', type=int, default=0)
    args, unknown = parser.parse_known_args()  # ignore unrecognized arguments

    main(args)
