# MealSAM_food_annotator
A semi-automatic segmentation tool for meal image segmentation using SAM in python=3.9.15
![Inputs & Outputs](/images/paperdigest_seg.png "Inputs & Outputs")

## MealSAM (ViT-B) vs. Pre-trained SAM models
![IoU](/images/heatmap.png "IoU")


## Installation

Follow these steps to set up the Annotation Tool environment:

### Step 0: Clone repository 

  

### Step 1: Create Conda Environment

1. Open your terminal or command prompt.

2. Create a new conda environment specifically for the Segmentation Tool V2 to manage dependencies efficiently by running the following command:

   ```bash
   conda create -n MealSAM python=3.9

### Step 2: Activate Conda Environment

After creating the environment, you need to activate it. Run the following command:
 conda activate MealSAM

### Step 3:  Install Required Packages
     
     pip install -r requirements.txt


## Using the Tool

### Step 1: Launching tool
From the repository, start the tool using Python.

  python ./MealSAM_food_annotator.py

### Step 2: Uploading Image
#### i.	Uploading an Image for segmentation
Note: Images are automatically resized. Upon saving, both the resized images and their corresponding validated masks of the same shape will be saved.



#### ii.	Visualizing Automatic Masks
Click on the "Segment" button to view all masks generated automatically - this is possible only with the pre-trained models. (This is only for visualization purposes; we are interested only in the semi-automatic segmentation and annotation of different food items or food containers present in the image). As you can see you have the option of changing the model used; mealSAM, base, large, huge.

E.g., ViT B

![ViT-B](/images/vit-B_automatic.png "ViT-B")


E.g., ViT L

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

