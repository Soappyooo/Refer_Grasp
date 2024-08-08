# Dataset Usage
## 1. File Structure
```
refgrasp
├─ train
│  ├─ depth
│  │  ├─ depth_00000000.png
│  │  └─ ...
│  ├─ mask
│  │  ├─ mask_00000000.png
│  │  └─ ...
│  ├─ rgb
│  │  ├─ rgb_00000000.jpg
│  │  └─ ...
│  ├─ expressions.json
│  └─ train.tsv
├─ test
│  └─ ...
└─ val
   └─ ...
```
## 2. JSON Structure
`expressions.json` is an array with info of multiple scenes. In each scene, we render multiple images without changing objects and expressions. In this json, each item has key `img_idxs` and `expressions`.  
The value of `img_idxs` is an array indicating the indices of images, e.g. `1` refers to `depth/depth_00000001.png`, `mask/mask_00000001.png` and `rgb/rgb_00000001.jpg`.  
The value of `expressions` is an array containing the referring expressions and objects refered to. For example, `["expressions"][0]["obj"]` has an `id` and `scene_id`. `id` is the id of the object's 3D model, **`scene_id` is the mask index of the object in mask images**. Unlike many other datasets, we use `255` to refer to the background of mask and `0`,`1`,`2`,... as the objects.
```
[
    {
        "img_idxs": [
            0,
            1,
            2,
            3,
            4
        ],
        "expressions": [
            {
                "obj": {
                    "id": "000017_thu_models_wooden_cup2",
                    "scene_id": 0
                },
                "expression": "Grab me the wooden cup that is not brown."
            },
            {
                "obj": {
                    "id": "000036_hope_models_OrangeJuice",
                    "scene_id": 2
                },
                "expression": "..."
            },
            ...
        ]
    },
    ...
]
```

## 3. TSV Structure
The tsv file has a structure of:
```
index \t path_to_image \t boundbox \t expression \t polygon \t polygon_interpolated
```
For example:
```
0	rgb/rgb_00000000.jpg	116,103,165,182 I would like to get the coca cola that is in the left rear.	119,109,121,107,... 119,109,120,108,...
```
`boundbox` has the format of `x1,y1,x2,y2` (the upper left point and lower right point), `polygon` and `polygon_interpolated` has the format of `x1,y1,x2,y2,x3,y3,...` with the point starting from minimal `x[i]**2 + y[i]**2` with a clockwise direction.  
The tsv file can be directly used for training models that predicts next point, or you can use `polygon_interpolated` or the json file to generate mask for mask prediction. For more information, see function `generate_tsv_file_for_RES()` in [./utils/dataset_utils.py](./utils/dataset_utils.py).

## 4. Dataset Visualization
Run [dataset_vis.py](./tool_scripts/dataset_vis.py) to visualize the dataset.

## 5. Depth Image Format
The format is float16, measured in meters.