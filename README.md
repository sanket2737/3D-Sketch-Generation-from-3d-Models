# 3D-Sketch-Generation-from-3d-Models

For the dissertation, i have used ShapeNetCore dataset.

------ created new render_models.py

 Command  :  xvfb-run -a blender --background --python render_models.py --model_name chair --num_sketches 1

imports model and render it using blender API and create three views - front view, 30 degree view and side view.

Then, existing code of CLIPasso - https://github.com/yael-vinker/CLIPasso, I have generated 2D sketches of the generated views of the model.

--- created new 2d_sketch_to_3d_sketch.py

It will fetches front and side sketch and performs operation to generate 3D sketch.

----------------------

Changes made in existing code files :

-- painterly_rendering.py

This Code added to store the final sketch at desired folder to fetch it to generate 3D sketch.

# Check if it's the last epoch
        if epoch == args.num_iter - 1:

            abs_path = os.path.abspath(os.getcwd())

            target = f"{abs_path}/Final_sketches"

            words_to_check = ['front', '30degree', 'side']

            # Find the word that matches in args.output_dir
            found_word = next((word for word in words_to_check if word in args.output_dir), None)

            desired_location = f"{abs_path}/Final_sketches"  # Replace with your path
            utils.plot_batch(inputs, sketches, desired_location, counter,
                             use_wandb=args.use_wandb, title=f"{found_word}_sketch.jpg")

