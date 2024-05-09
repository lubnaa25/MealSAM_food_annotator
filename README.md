# MealSAM_food_annotator
## License

Copyright © 2024 University of Bern, ARTORG Center for Biomedical Engineering Research, [Lubnaa Abdur Rahman, Ioannis Papathanail, Lorenzo Brigato, Stavroula Mougiakakou]

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)


## A SAM-based Tool for Semi-Automatic Food Annotation.

This page features the code for the semi-automatic segmentation tool submitted for demo presentation pending acceptance.
The tool is designed for meal image segmentation leveraging [SAM](https://github.com/facebookresearch/segment-anything) featuring pre-trained versions of SAM along with a fine-tuned version of SAM's mask decoder, dubbed MealSAM, with the ViT-B backbone tailored specifically for food image segmentation.

<!---![Inputs & Outputs](/images/paperdigest_seg.png "Inputs & Outputs")--->
<p align="center">
<img src="/images/paperdigest_seg.png" width=50% height=50%>
</p>


## MealSAM (ViT-B) vs. Pre-trained SAM models
<!---![IoU](/images/heatmap.png "IoU")--->
<p align="center">
<img src="/images/heatmap.png" width=50% height=50%>
</p>

## Structure
```bash
     MealSAM
     ├── images
     ├── tool_resources
     │   └── appicon.json
     │   └── categories.json  -- can be changed to your categories
     │   └── save.png
     │   └── upload.png
     ├── weights
     │   ├── MealSAM.pth -- can be changed to your model checkpoint
     │   └── sam_vit_b_01ec64.pth
     │   └── sam_vit_l_0b3195.pth
     │   └── sam_vit_h_4b8939.pth
     ├── MealSAM_food_annotator.py
     ├── README.md
     ├── requirements.txt
     └── requirements_cuda.txt
```

## Installation

Follow these steps to set up the Annotation Tool environment:

### Step 0: Clone repository 


```ruby
     git clone https://github.com/lubnaa25/MealSAM_food_annotator.git
   
     cd MealSAM_food_annotator
```  

### Step 1: Create Conda Environment

1. Open your terminal or command prompt.

2. Create a new conda environment specifically for the Segmentation Tool V2 to manage dependencies efficiently by running the following command:

```ruby
   conda create -n MealSAM python=3.9
```

### Step 2: Activate Conda Environment

After creating the environment, you need to activate it. Run the following command:
     
```ruby
     conda activate MealSAM
```

### Step 3:  Install Required Packages
     
```ruby
     pip install -r requirements_cuda.txt (or requirements.txt for non GPU)
```

### Step 4: Download the checkpoints 
Put the checkpoints in the ./weights folder

Pre-trained SAM can be downloaded from [official SAM repo](https://github.com/facebookresearch/segment-anything)

MealSAM can be downloaded [here](https://www.dropbox.com/scl/fi/o41lkdu7wacyosurmr7dk/MealSAM.pth?rlkey=fe2df1k4hic80uztk54zd7u7q&st=db5r1c99&dl=0)

## Using the Tool

### Step 1: Launching tool
From the repository, start the tool using Python.

```ruby
    python ./MealSAM_food_annotator.py
```

### Step 2: Uploading Image
#### i.	Uploading an Image for segmentation
Note: Images are automatically resized. Upon saving, both the resized images and their corresponding validated masks of the same shape will be saved.



#### ii.	Visualizing Automatic Masks
Click on the "Segment" button to view all masks generated automatically - this is possible only with the pre-trained models. (This is only for visualization purposes; we are interested only in the semi-automatic segmentation and annotation of different food items or food containers present in the image). As you can see you have the option of changing the model used; mealSAM, base, large, huge.

E.g., ViT-B

![ViT-B](/images/vit-B_automatic.png "ViT-B")


E.g., ViT-L

![ViT-H](/images/vit-L_automatic.png "ViT-L")


#### iii.	Semi-Automatic Segmentation
Perform semi-automatic segmentation by interacting with the image:
•	Left-click on pixels to include them (up to 10 points).
•	Right-click on pixels to exclude them (up to 10 points).
Click on the "Semi Segment" button to generate a semi-automatic mask. This produces a mono mask (only one mask is generated).

![Semi-auto](/images/annotatedexampleMealSAM.png "Semi Auto")


#### iv.	Validating the Mask
If the semi-automatic mask is satisfactory, assign a category to it and click "Validate."
This step confirms the category for the segmented area. In the case the category is not present in the drop down list as you type, you can add new category (select Add new category, Press ENTER, and fill in the pop-up)


#### v.	Repeating Segmentation for Different Categories
To segment other items, first click "Clear" to remove the inclusion and exclusion points.
Repeat the segmentation process and assign different categories as needed.
Before clicking "Validate," ensure you reselect the category. 
As you can see below, here you can input either the weight/volume if you know this information. This step is fully optional.


#### vi. Save 
Once satisfied with the segmentation, click "Save."
The final output will be the validated mask. Upon saving, both the resized RGB image and the validated mask (saved as a 16-bit image) are stored.


## Coming soon 
In the future, we anticipate support of bounding boxes as prompts in the tool and also the release of larger versions of MealSAM.


## Beyond Food Image annotation
The tool can be extented for your own use case since we also include the pre-trained versions of SAM. You can also replace MealSAM by your fine-tuned version of SAM within the tool and switch the categories.json file to your list of categories.
Lines of code in MealSAM_food_annotator.py to be changed:
```ruby
     MealSAM weight:   lines 207, 421, 435 
     Categories:       lines 289, 465
```

## Citation

If you find either MealSAM or this tool useful, please consider citing it using the following BibTeX entry:

```bibtex
@software{abdurrahman2024mealsamfoodannotator,
  author = {Lubnaa Abdur Rahman, Ioannis Papathanail, Lorenzo Brigato, Stavroula Mougiakakou},
  title = {{A SAM-based Tool for Semi-Automatic Food Annotation}},
  url = {https://github.com/lubnaa25/MealSAM_food_annotator},
  version = {1.0.0},
  year = {2024}
}
