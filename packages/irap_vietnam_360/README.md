# irap-vietnam-360

**NOTE: This project has been superseded by https://github.com/Ivan1248/gyroflow**

This project converts fisheye video sequences to perspective projection videos or images using GPS track data, saving one image every distance step along the path.

## Usage

```sh
python -m irap_vietnam_360.generate_image_dataset input output --distance_step 10 --fov_h 127 --output_width 384 --output_height 288 --pitch 12
```

- `input`: Root directory containing subfolders, each with a fisheye video (`.insv`) and a corresponding GPS track file (`.gpx`).
- `output`: Directory where extracted images will be saved, mirroring the input folder structure.
- `--distance_step`: Distance in meters between extracted frames (default: 10).
- `--fov_h`: Horizontal field of view for the perspective output, in degrees (default: 127).
- `--output_width`: Output image width in pixels (default: 384).
- `--output_height`: Output image height in pixels (default: 288).
- `--yaw`: Horizontal rotation angle (in degrees) of the virtual camera (default: 0.0).
- `--pitch`: Vertical rotation angle (in degrees) of the virtual camera; positive tilts the view upward (default: 0.0).
- `--roll`: Optical axis rotation angle (in degrees) of the virtual camera (default: 0.0).
- `--fisheye_radius_factor`: Fisheye radius as a fraction of the image radius, for tuning the effective fisheye circle (default: 0.94).

### Input directory structure

```
input/
  ├── 20231223_unit1_58_59_836/
  │     ├── VID_20241219_105926_00_145.insv   # Fisheye video file
  │     ├── activity_17792202410.gpx          # GPS track file
  │     └── ... (other files)
  ├── 20241210_unit1_32/
  │     └── ...
  └── ...
```

Each subfolder represents a sequence and must contain one `.insv` video and one `.gpx` file.

### Output directory structure

```
output/
  ├── 20231223_unit1_58_59_836/
  │     ├── 0000000.png   # Extracted perspective images
  │     ├── 0000127.png
  │     └── ...
  ├── 20241210_unit1_32/
  │     └── ...
  └── ...
```

Each output subfolder matches the input and contains PNG images named by their frame index, corresponding to the extracted frames at each distance step.

## To do

- The synchronization between video duration and GPS track duration is currently handled by a simple rescaling hack. This should be improved for better alignment.
- The camera usually does not point exactly in the movement direction and sometimes points backwards. Add recognition of the road direction or the movement direction (including whether the camera points backward) or use gyroscope measurements?
  - The synchronization of frame orientations with frames might not be perfect (there is a few more orientations than frames).
  - Subtraction of estimated orientations does not work correctly.
- Check anti-aliasing and interpolation.
- Check the perspective conversion code.
