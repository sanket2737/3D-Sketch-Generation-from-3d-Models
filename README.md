# 3D-Sketch-Generation-from-3d-Models

For the dissertation, i have used ShapeNetCore dataset.

------ created new render_models.py

 Command  :  xvfb-run -a blender --background --python render_models.py --model_name chair --num_sketches 1

imports model and render it using blender API and create three views - front view, 30 degree view and side view.

Then, existing code of CLIPasso - https://github.com/yael-vinker/CLIPasso, I have generated 2D sketches of the generated views of the model.

--- created new 2d_sketch_to_3d_sketch.py

It will fetches front and side sketch and performs operation to generate 3D sketch.


