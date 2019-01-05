# Truncation LINEMOD

## Dataset parameters

* Objects: 13
* Object models: Mesh models with surface color and normals.
* Training images: None. **This dataset is only for testing.**
* Testing images: We took all images in the LINEMOD dataset [1], and cropped them to [256, 256] images where the target object remains only 40% to 60% visible area.

## Dataset format

### Directory structure

The datasets have the following structure:

* TRUNCATION_LINEMOD.md - Dataset-specific information.
* MODELTYPE - Testing images and ground truth.
* models - 3D object models that correspond to the ground-truth 6D poses.

### MODELTYPE directory

Each object model has its testing images and ground truth in its own MODELTYPE directory. The testing data in the directory is organized as:

* {:06d}_rgb.jpg - Color images.
* {:06d}_info.pkl - 6D pose and camera intrinsic matrix.
* {:04d}_msk.png - Masks of the regions of interest.

Note that, the `{:06d}_info.pkl` is a pickle file containing a python dict, which can be read by the following function

```python
import pickle

def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
```

## References

[1] Hinterstoisser et al. "Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes" ACCV 2012.

