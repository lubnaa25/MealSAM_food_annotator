# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:16:30 2024

@author: Lubnaa Abdur Rahman


Copyright [2024] [Lubnaa Abdur Rahman, Ioannis Papathanail, Lorenzo Brigato, Stavroula Mougiakakou]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, Menu
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import json
#os.chdir('./MealSAM_food_annotator/') ##Replace absolute path

from segment_anything import SamAutomaticMaskGenerator, SamPredictor

from functools import partial
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )

def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "MealSAM": build_sam_vit_b,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}
def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if checkpoint is not None:
        if device==torch.device('cpu'):
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=torch.device('cpu'))
        else:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


#%%
def update_sam_model(model_type, sam_checkpoint):
    global sam  
    device = "cpu" if model_type in ["vit_l", "vit_h"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)

class ScrolledListbox(tk.Toplevel):
    def __init__(self, parent, options, var, **kwargs):
        super().__init__(parent)
        self.var = var
        self.listbox = tk.Listbox(self, **kwargs)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y) 
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        for option in options:
            self.listbox.insert(tk.END, option)
        self.listbox.bind("<<ListboxSelect>>", self.selected)

    def selected(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            value = event.widget.get(index)
            self.var.set(value)
            self.destroy()


class AutocompleteCombobox(ttk.Combobox):
    def __init__(self, parent, categories, app_instance, **kwargs):
        super().__init__(parent, **kwargs)
        self.app_instance = app_instance
        self.categories = categories
        self["values"] = categories
        self.bind("<KeyRelease>", self.on_keyrelease)
        longest_category = max(categories, key=len)
        self.config(width=len(longest_category) + 3)  

    
    def on_keyrelease(self, event):
        if event.keysym in ["Up", "Down", "Left", "Right", "Return", "Tab", "Escape"]:
            if event.keysym == "Return" and self.get() == "Add new category...":
                self.app_instance.add_new_category() 
            return

        if event.keysym == "Escape":
            self.event_generate("<Escape>")
            return
     
        value = event.widget.get().strip()
        if value:
            filtered_data = [item for item in self.categories if value.lower() in item.lower()]
            if "Add new category..." not in filtered_data:
                filtered_data.append("Add new category...")  
            self["values"] = filtered_data
        else:
            self["values"] = self.categories + ["Add new category..."]  
    
      
        self.event_generate("<Down>")
        self.icursor(tk.END)
        

        

class ImageEditorApp:
    def __init__(self, root):
      
        self.root = root
        self.model_type = "MealSAM"  # Default model type
        self.sam_checkpoint = "./weights/MealSAM.pth"
        self.action_history=[]
        
        
        
        self.model_variable = tk.StringVar(value="MealSAM")  
        self.mask_generator = self.update_model_selection()  

     
        model_options = ["MealSAM","vit_b", "vit_l", "vit_h"]
       
        
        self.root.title("Food Annotator")
        icon = ImageTk.PhotoImage(file="./tool_resources/appicon.png")  
        self.root.iconphoto(False, icon)
       
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        self.image_uploaded = False
        
        #File menu 
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Upload", command=self.upload_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
     
        upload_image_icon = ImageTk.PhotoImage(Image.open("./tool_resources/upload.png").resize((20, 20)))
        save_image_icon = ImageTk.PhotoImage(Image.open("./tool_resources/save.png").resize((20, 20)))
        
        ### Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", anchor="n", pady=5)
        
        #Upload Image
        self.upload_button = tk.Button(button_frame, text="Upload", image=upload_image_icon, compound="left",
                                       command=self.upload_image)
        self.upload_button.image = upload_image_icon
        self.upload_button.pack(side="left", padx=5)
        
        
        #Save Files - Save Validated Mask, and resized image
        self.save_button = tk.Button(button_frame, text="Save", image=save_image_icon, compound="left",
                                     command=self.save_image)
        self.save_button.image = save_image_icon
        self.save_button.pack(side="left", padx=5)
        
        
        model_selection_label = tk.Label(button_frame, text="Select Model:")
        model_selection_label.pack(side="left", padx=5)

 
        self.model_selection_dropdown = ttk.Combobox(button_frame, textvariable=self.model_variable, values=model_options, state="readonly")
        self.model_selection_dropdown.pack(side="left", padx=5)
        self.model_selection_dropdown.bind("<<ComboboxSelected>>", self.update_model_selection)

        
        
        self.mask_generator = update_sam_model(self.model_variable.get(), self.determine_checkpoint_path(self.model_variable.get()))
        
       
          
        
        #Clears the include and exclude points 
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_points)
        self.clear_button.pack(side="left", padx=5)
        
        #Undo previously clicked point on the RGB
        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo_point)
        self.undo_button.pack(side="left", padx=5)
        
        #Provided include/exclude points Mono Mask generated for object         
        self.semi_segment_button = tk.Button(button_frame, text="Semi-Segment", command=self.semi_segment)
        self.semi_segment_button.pack(side="left", padx=5)
        
        
                     
        #food cat/labels to assign per specific segment       
        self.category_variable = tk.StringVar(root)
        self.load_categories_from_json("./tool_resources/categories.json")
        self.category_dropdown_label = tk.Label(button_frame, text="Select Segment Category:")
        self.category_dropdown_label.pack(side="left", padx=5)
        self.category_dropdown = self.category_dropdown = AutocompleteCombobox(button_frame, self.categories, self, textvariable=self.category_variable)
        self.category_dropdown.pack(side="left", padx=5)
        style = ttk.Style(self.root)
        style.theme_use("alt")  
        style.configure("TMenubutton", background="white", foreground="black")
        style.map("TMenubutton", background=[("active", "grey")])  
        
        self.annotation_option = tk.StringVar(value="No")  
        self.annotation_type = tk.StringVar(value="Weight")  
        
       
        self.annotation_option_label = tk.Label(button_frame, text="Input weight/volume?")
        self.annotation_option_label.pack(side="left", padx=5)
        self.yes_radio_button = tk.Radiobutton(button_frame, text="Yes", variable=self.annotation_option, value="Yes", command=self.toggle_annotation_fields)
        self.no_radio_button = tk.Radiobutton(button_frame, text="No", variable=self.annotation_option, value="No", command=self.toggle_annotation_fields)
        self.yes_radio_button.pack(side="left", padx=5)
        self.no_radio_button.pack(side="left", padx=5)
        
        
        self.annotation_type_label = tk.Label(button_frame, text="Type:", state="disabled")
        self.annotation_type_menu = ttk.Combobox(button_frame, textvariable=self.annotation_type, state="disabled", values=["Weight", "Volume"])
        self.annotation_type_label.pack(side="left", padx=5)
        self.annotation_type_menu.pack(side="left", padx=5)

        
        
        self.grams_label = tk.Label(button_frame, text="Weight (g) / Volume (ml):")
        self.grams_label.pack(side="left", padx=10)
        
        self.grams_entry = tk.Entry(button_frame)  
        self.grams_entry.pack(side="left", padx=10)

    
        self.yes_radio_button = tk.Radiobutton(button_frame, text="Yes", variable=self.annotation_option, value="Yes", command=self.toggle_annotation_fields)
        self.no_radio_button = tk.Radiobutton(button_frame, text="No", variable=self.annotation_option, value="No", command=self.toggle_annotation_fields)
       
        
        
        self.validate_mask_button = tk.Button(button_frame, text="Validate", command=self.validate_mask)
        self.validate_mask_button.pack(side="left", padx=10)
       
      
        self.image_on_canvas = None
        self.image_path = None
        self.photo_image = None
        self.semi_segmented_mask=None
        self.mask_image_on_canvas = None
        self.mask_photo_image = None
        self.overlaid_mask_canvas = None
        self.overlaid_validated_mask_canvas=None
        self.include_pixels = []
        self.exclude_pixels = []
        self.include_click_count = 0
        self.exclude_click_count = 0
        self.image_directory=""
        self.display_label=[]
        self.segment_data = {}  
        self.all_nutrient_data = []
        
        
        
        
        #Include/Exclude points
        self.include_label = tk.Label(self.root, text="Include Pixels: ")
        self.include_label.pack(side="bottom")
        self.exclude_label = tk.Label(self.root, text="Exclude Pixels: ")
        self.exclude_label.pack(side="bottom")
        
        self.display_label = tk.Label(self.root, text="Category and weight: ")
        self.display_label.pack(side="bottom")
        
        #Clears all the canvas items except the RGB image
        self.clear_all_button = tk.Button(button_frame, text="Clear All", command=self.clear_canvas_all)
        self.clear_all_button.pack(side="left", padx=5)
        
        
        #Will produce the automatic segmentation mask from SAM
        self.segment_button = tk.Button(button_frame, text="Segment", command=self.segment_image)
        self.segment_button.pack(side="left", padx=5)
        
        ###Displaying of imgs and masks
        
        #RGB Image
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True, side="left")
        self.rgb_image_label = tk.Label(self.canvas_frame, text="RGB Image")
        self.rgb_image_label.pack()
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        
        #Colored mask generated by SAM either by clicking "Segment" or "Semi-segment" buttons
        self.mask_canvas_frame = tk.Frame(self.root)
        self.mask_canvas_frame.pack(fill="both", expand=True, side="left")
        self.mask_label = tk.Label(self.mask_canvas_frame, text="Model's Mask")
        self.mask_label.pack()
        self.mask_canvas = tk.Canvas(self.mask_canvas_frame, cursor="cross")
        self.mask_canvas.pack(fill="both", expand=True)

        #Overlaid mask generated by SAM on top of the RGB        
        self.overlaid_mask_frame = tk.Frame(self.root)
        self.overlaid_mask_frame.pack(fill="both", expand=True, side="left")
        self.overlaid_mask_label = tk.Label(self.overlaid_mask_frame, text="Overlaid Model's mask")
        self.overlaid_mask_label.pack()
        self.overlaid_mask_canvas = tk.Canvas(self.overlaid_mask_frame, cursor="cross")
        self.overlaid_mask_canvas.pack(fill="both", expand=True)
        
        #Colored validated mask - can contain more than one segmented object - Validate button must be clicked 
        self.validated_mask_frame = tk.Frame(self.root)
        self.validated_mask_frame.pack(fill="both", expand=True, side="left")
        self.validated_mask_label = tk.Label(self.validated_mask_frame, text="Validated Mask")
        self.validated_mask_label.pack()
        self.validated_mask_canvas = tk.Canvas(self.validated_mask_frame, cursor="cross")
        self.validated_mask_canvas.pack(fill="both", expand=True)
     
        #Overlaid validated mask on the RGB
        self.overlaid_validated_mask_frame = tk.Frame(self.root)
        self.overlaid_validated_mask_frame.pack(fill="both", expand=True, side="right")
        self.overlaid_validated_mask_label = tk.Label(self.overlaid_validated_mask_frame, text="Overlaid Validated Mask")
        self.overlaid_validated_mask_label.pack()
        self.overlaid_validated_mask_canvas = tk.Canvas(self.overlaid_validated_mask_frame, cursor="cross")
        self.overlaid_validated_mask_canvas.pack(fill="both", expand=True)

    


        
    def determine_checkpoint_path(self, model_type):
        if model_type == "MealSAM":
            return "./weights/MealSAM.pth"
          
        elif model_type == "vit_b":
            return "./weights/sam_vit_b_01ec64.pth"
        
        elif model_type == "vit_l":
            return "./weights/sam_vit_l_0b3195.pth"
        elif model_type == "vit_h":
            return "./weights/sam_vit_h_4b8939.pth"

    def update_model_selection(self, event=None):
        self.model_type = self.model_variable.get()  # Get the current selection from the dropdown
        if self.model_type == "MealSAM":
            self.model_type ="vit_b"
            self.sam_checkpoint = "./weights/MealSAM.pth"
         
         
        elif self.model_type == "vit_b":
           
            self.sam_checkpoint = "./weights/sam_vit_b_01ec64.pth"
        elif self.model_type == "vit_l":
            self.sam_checkpoint = "./weights/sam_vit_l_0b3195.pth"
        elif self.model_type == "vit_h":
            self.sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"

        # Initialize SAM model with current selection
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        device = "cpu" if self.model_type in ["vit_l", "vit_h"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    
    def add_new_category(self):
        new_category_name = tk.simpledialog.askstring("New Category", "Enter new category name:")
        if new_category_name and new_category_name.strip() and new_category_name not in self.categories:
            self.categories.append(new_category_name)
            self.categories.sort()  # .
            self.category_dropdown["values"] = self.categories + ["Add new category..."]
            self.update_categories_json()  # Update the JSON file with the new category.
            self.category_variable.set(new_category_name)  # Set the newly added category as the selected value.
    
    def update_categories_json(self):
        # 
        try:
            with open("./tool_resources/categories.json", "w") as file:
                json.dump(self.categories, file)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to update categories: {e}")

    def toggle_annotation_fields(self):
        if self.annotation_option.get() == "Yes":
            #  annotation type selection (weight/volume)
            self.annotation_type_label.config(state="normal")
            self.annotation_type_menu.config(state="readonly") 
            self.grams_label.config(state="normal")
            self.grams_entry.config(state="normal")
        else:
            # disable
            self.annotation_type_label.config(state="disabled")
            self.annotation_type_menu.config(state="disabled")
            self.grams_label.config(state="disabled")
            self.grams_entry.config(state="disabled")
            self.grams_entry.delete(0, tk.END)  

    
    def open_scrolled_listbox(self):
        ScrolledListbox(self.root, self.categories, self.category_variable, height=10)
        

    def load_categories_from_json(self, json_file_path):
        try:
            with open(json_file_path, "r") as file:
                data = json.load(file)
             
                
            self.categories = data

            
            self.color_map = self.create_color_map()
        except FileNotFoundError:
            print(f"The file {json_file_path} was not found.")
            self.categories = []
        except json.JSONDecodeError:
            print(f"The file {json_file_path} does not contain valid JSON.")
            self.categories = []
            
    #Logic for Upload button
    def upload_image(self):
        self.include_pixels = []
        self.exclude_pixels = []
        self.all_nutrient_data = []  
        self.update_nutrient_data_display()
        
        self.include_click_count = 0
        self.exclude_click_count = 0
    
        self.include_label.config(text="Include Pixels (x,y): ") #Include pixels are assigned a label of 1 - i.e. Foreground -- appear blue
        self.exclude_label.config(text="Exclude Pixels (x,y): ") #Exclude pixels are assigned a label of 0 - i.e. Background -- appear pink
        
        if hasattr(self, "validated_mask"):
            del self.validated_mask
            
        if hasattr(self, "val_copy"):
            del self.val_copy
        
        filetypes = [("JPG files", "*.jpg"), ("PNG files", "*.png")]
        self.image_path = filedialog.askopenfilename(filetypes=filetypes)
        if not self.image_path:
            return
        
        #save the directory for proper file saving later in same directory as previously
        self.image_directory = os.path.dirname(self.image_path) 
        self.image_filename = os.path.basename(self.image_path)
        
        self.image_uploaded = True
        original_image = Image.open(self.image_path)
        
        if original_image.width != 380:
            new_height = int(original_image.height * (380 / original_image.width))
            original_image = original_image.resize((380, new_height))
            
        
        self.image = original_image
        self.photo_image = ImageTk.PhotoImage(self.image)
        
        self.overlaid_mask_canvas.delete("all")
        self.validated_mask_canvas.delete("all")
        self.overlaid_validated_mask_canvas.delete("all")
    
        self.create_mask()
        self.reset_canvas()
        self.mask_canvas.delete("all")
        
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor="nw", image=self.photo_image)
    
        self.canvas.image = self.photo_image
    
        self.canvas.bind("<Button-1>", self.include_left_click)
        self.canvas.bind("<Button-3>", self.exclude_right_click)

    def save_image(self):
        if hasattr(self,"image_filename"):
            original_filename, original_extension = os.path.splitext(self.image_filename)
        
        if self.image_path:
            if hasattr(self, "val_copy"):
                
                validated_mask_image = self.val_copy
                val_filename = f"{original_filename}_validated_mask.png"
                val_path = os.path.join(self.image_directory, val_filename)
                cv2.imwrite(val_path, validated_mask_image)
                output_filename = f"{original_filename}_resized.png"
                output_path = os.path.join(self.image_directory, output_filename)
                self.image.save(output_path)
                
                annotated_weight_filename = f"{original_filename}_weights_info.txt"
                nutrient_info_path = os.path.join(self.image_directory, annotated_weight_filename)
                with open(nutrient_info_path, 'w') as nutrient_file:
                    for entry in self.all_nutrient_data:
                        weight = entry.get('Weight')  
                        volume = entry.get('Volume')  
                        if weight:
                            nutrient_file.write(f"{entry['Category']}: {weight} grams\n")
                        elif volume:
                            nutrient_file.write(f"{entry['Category']}: {volume} ml\n")
                        else:
                            nutrient_file.write(f"{entry['Category']}: No weight/volume specified\n")
                
                tk.messagebox.showinfo("Success", "Image, validated mask, and weight information saved successfully!")
            else:
                
                tk.messagebox.showerror("Error", "Nothing available to be saved. Both Image and Validated mask must be present")
        else:
           
            tk.messagebox.showerror("Error", "Nothing available to be saved. Both Image and Validated mask must be present")

    def create_mask(self):
        mask = np.zeros((self.image.height, self.image.width), dtype=np.uint8)
        mask_image = Image.fromarray(mask)

        self.mask_photo_image = ImageTk.PhotoImage(mask_image)
        if self.mask_image_on_canvas is not None:
            self.mask_canvas.delete(self.mask_image_on_canvas)  
        
        self.mask_image_on_canvas = self.mask_canvas.create_image(
            0, 0, anchor="nw", image=self.mask_photo_image)
 
    #Used by clear all button   
    def clear_canvas_all(self):
        self.include_pixels = []
        self.exclude_pixels = []
        self.include_click_count = 0
        self.exclude_click_count = 0
        self.update_labels()
        self.action_history = []  
        self.all_nutrient_data = []  
        self.update_nutrient_data_display()
        self.canvas.delete("highlighted_pixel")
        self.overlaid_validated_mask_canvas.delete("all")
        self.validated_mask_canvas.delete("all")
        self.overlaid_mask_canvas.delete("all")
        self.mask_canvas.delete("all")
        self.category_variable.set("")
        self.validate_mask=None
        self.validated_mask= None
       

   
    def segment_image(self):
        if not self.image_uploaded:
            tk.messagebox.showerror("Error", "Please upload an image first.")
            return
        
        original_image_array = np.array(self.image.convert("RGB"))  
    
        image_array = np.asarray(self.image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        masks = self.mask_generator.generate(image_array)
    
        num_masks = len(masks)
        merged_mask = np.zeros_like(masks[0]["segmentation"], dtype=int)
    
        colors = np.random.randint(0, 256, size=(num_masks, 3), dtype=np.uint8)
    
        for i in range(num_masks):
            mask_value = i + 1
            merged_mask[masks[i]["segmentation"]] = mask_value
    
        color_mask = np.zeros((merged_mask.shape[0], merged_mask.shape[1], 3), dtype=np.uint8)
        for i in range(num_masks):
            color_mask[merged_mask == i + 1] = colors[i]
    
        alpha = 0.5 
        mask_indices = np.any(color_mask != [0, 0, 0], axis=-1)  
        overlay_image_array = original_image_array.copy()
        overlay_image_array[mask_indices] = (alpha * color_mask[mask_indices] + (1 - alpha) * original_image_array[mask_indices]).astype("uint8")
        
        overlaid_image_array = original_image_array.copy()
        overlaid_image_array[mask_indices] = (alpha * color_mask[mask_indices] + (1 - alpha) * original_image_array[mask_indices]).astype("uint8")
        overlaid_image = Image.fromarray(overlaid_image_array)
        overlaid_photo_image = ImageTk.PhotoImage(overlaid_image)
        
        self.overlaid_mask_canvas.delete("all") 
        self.overlaid_mask_canvas.create_image(0, 0, anchor="nw", image=overlaid_photo_image)
        self.overlaid_mask_canvas.image = overlaid_photo_image
      
        new_mask_photo_image = ImageTk.PhotoImage(Image.fromarray(color_mask))
        self.mask_canvas.delete("all")  
        self.mask_image_on_canvas = self.mask_canvas.create_image(
            0, 0, anchor="nw", image=new_mask_photo_image)
        self.mask_canvas.image = new_mask_photo_image 
                

    #unique color for each single category
    def create_color_map(self):
        num_categories = len(self.categories)
        num_categories +=1
        color_map = np.zeros((num_categories, 3), dtype=np.uint16)
        for i in range(num_categories):
            color_map[i] = [i * 20, i * 30, i * 50]
        return color_map

    #Semi-segment button logic - considers include/exclude points
    def semi_segment(self):
        if not self.image_uploaded:
            tk.messagebox.showerror("Error", "Please upload an image first.")
            return
        
        if not self.include_pixels:
            tk.messagebox.showerror("Error", "Semi Segment needs include points")
            return
    
        if app.exclude_pixels:
            include_coords = np.asarray(app.include_pixels)
            exclude_coords = np.asarray(app.exclude_pixels)
            include_labels = np.array([1] * len(app.include_pixels)) #Include is foreground
            exclude_labels = np.array([0] * len(app.exclude_pixels)) #Exclude is background
            inputarray = np.concatenate((include_coords, exclude_coords))
            input_label = np.concatenate((include_labels, exclude_labels))
        else:
            inputarray = np.asarray(self.include_pixels)
            input_label = np.array([1] * len(app.include_pixels))
    
        #Call model - multimasks not to be generated
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        device="cpu" if self.model_type in ["vit_l", "vit_h"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        sam.to(device=device)
        predictor = SamPredictor(sam)
        image_array = np.asarray(self.image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_array)
        masks, scores, logits = predictor.predict(
            point_coords=inputarray,
            point_labels=input_label,
            multimask_output=False,
        )
     
        num_masks = len(masks)
        merged_mask = np.zeros_like(masks[0], dtype=int)

        colors = np.random.randint(0, 256, size=(num_masks, 3), dtype=np.uint8)

        for i in range(num_masks):
            mask_value = i + 1
            merged_mask[masks[i] == mask_value] = mask_value
    
        self.merged_mask=merged_mask
        color_mask = np.zeros((merged_mask.shape[0], merged_mask.shape[1], 3), dtype=np.uint8)
        for i in range(num_masks):
            color_mask[merged_mask == i + 1] = colors[i]
    
        mask_indices = np.any(color_mask != [0, 0, 0], axis=-1)
        overlay_image_array = image_array.copy()
        overlay_image_array=cv2.cvtColor(overlay_image_array, cv2.COLOR_BGR2RGB)
        alpha = 0.5
        overlay_image_array[mask_indices] = (
            alpha * color_mask[mask_indices] + (1 - alpha) * image_array[mask_indices]
        ).astype("uint8")
        
        self.semi_segmented_mask=color_mask
    
        overlaid_mask_image = Image.fromarray(overlay_image_array)
        overlaid_mask_photo_image = ImageTk.PhotoImage(overlaid_mask_image)
  
        self.overlaid_mask_canvas.delete("all")
        self.overlaid_mask_canvas.create_image(0, 0, anchor="nw", image=overlaid_mask_photo_image)
        self.overlaid_mask_canvas.image = overlaid_mask_photo_image
        
        new_mask_photo_image = ImageTk.PhotoImage(Image.fromarray(color_mask))
        self.mask_canvas.delete("all")  
        self.mask_image_on_canvas = self.mask_canvas.create_image(
            0, 0, anchor="nw", image=new_mask_photo_image)
        self.mask_canvas.image = new_mask_photo_image 
       
        
      
    def display_colored_validated_mask(self, validated_mask, color_map):    
            colored_validated_mask = np.zeros((validated_mask.shape[0], validated_mask.shape[1], 3), dtype=np.uint8)
            for category_index, color in enumerate(color_map):
                colored_validated_mask[validated_mask == category_index] = color
      
            colored_validated_mask_image = Image.fromarray(colored_validated_mask)
            colored_validated_mask_photo_image = ImageTk.PhotoImage(colored_validated_mask_image)   
            self.validated_mask_canvas.delete("all")
            self.validated_mask_canvas.create_image(0, 0, anchor="nw", image=colored_validated_mask_photo_image)
            self.validated_mask_canvas.image = colored_validated_mask_photo_image
        
    
    def display_colored_validated_mask2(self, validated_mask, color_map, alpha=0.5):
        colored_validated_mask = np.zeros((validated_mask.shape[0], validated_mask.shape[1], 3), dtype=np.uint8)
        for category_index, color in enumerate(color_map):
            colored_validated_mask[validated_mask == category_index] = color
    
        overlay_image_array = np.array(self.image.convert("RGB"))
        mask_indices = np.any(colored_validated_mask != [0, 0, 0], axis=-1)
        overlay_image_array[mask_indices] = (
            alpha * colored_validated_mask[mask_indices] + (1 - alpha) * overlay_image_array[mask_indices]
        ).astype("uint8")
    
        overlaid_validated_mask_image = Image.fromarray(overlay_image_array)
        overlaid_validated_mask_photo_image = ImageTk.PhotoImage(overlaid_validated_mask_image)
    
        self.overlaid_validated_mask_canvas.delete("all")
        self.overlaid_validated_mask_canvas.create_image(0, 0, anchor="nw", image=overlaid_validated_mask_photo_image)
        self.overlaid_validated_mask_canvas.image = overlaid_validated_mask_photo_image

    

    def validate_mask(self):
        if not self.image_uploaded:
            tk.messagebox.showerror("Error", "Please upload an image first.")
            return
    
        if self.semi_segmented_mask is None:
            tk.messagebox.showerror("Error", "No semi-segmented mask available.")
            return
    
        selected_category = self.category_variable.get()
        if not selected_category:
            tk.messagebox.showerror("Error", "No category selected.")
            return
    
       
        try:
            category_index = self.categories.index(selected_category) + 1
        except ValueError:
            tk.messagebox.showerror("Error", "Selected category not found in categories list.")
            return
    
     
        annotation_data = None
        if self.annotation_option.get() == "Yes":
            annotation_type = self.annotation_type.get()
            try:
                annotation_data = float(self.grams_entry.get())
            except ValueError:
                tk.messagebox.showerror("Error", f"Invalid input for {annotation_type.lower()}. Please enter a valid number.")
                return
    
        
        existing_entry = next((entry for entry in self.all_nutrient_data
                               if entry['Include Pixels'] == self.include_pixels and entry['Exclude Pixels'] == self.exclude_pixels), None)
    
        if existing_entry:
            existing_entry['Category'] = selected_category
            if annotation_data is not None:
                existing_entry[annotation_type] = annotation_data
        else:
            new_entry = {
                "Category": selected_category,
                "Include Pixels": self.include_pixels.copy(),
                "Exclude Pixels": self.exclude_pixels.copy(),
            }
            if annotation_data is not None:
                new_entry[annotation_type] = annotation_data
            self.all_nutrient_data.append(new_entry)
    
        # Merge masks w new category index
        if hasattr(self, "validated_mask"):
            existing_validated_mask = self.validated_mask
        else:
            existing_validated_mask = None
    
        validated_mask = self.merge_masks(self.merged_mask, existing_validated_mask, category_index)
        self.validated_mask = validated_mask
    
        self.val_copy = validated_mask.copy()
    
        validated_mask_image = Image.fromarray(validated_mask)
        validated_mask_photo_image = ImageTk.PhotoImage(validated_mask_image)
    
        self.validated_mask_canvas.delete("all")
        self.validated_mask_canvas.create_image(0, 0, anchor="nw", image=validated_mask_photo_image)
        self.validated_mask_canvas.image = validated_mask_photo_image
    
        self.display_colored_validated_mask(validated_mask, self.color_map)
        self.display_colored_validated_mask2(validated_mask, self.color_map)
    
        self.update_nutrient_data_display()
    
    

    
    
    def update_nutrient_data_display(self):
        display_text = "Nutrient Data:\n"
        for entry in self.all_nutrient_data:
            display_text += f"{entry['Category']}: "
            if 'Weight' in entry:
                display_text += f"{entry['Weight']} grams\n"
            elif 'Volume' in entry:
                display_text += f"{entry['Volume']} ml\n"
            else:
                display_text += "No weight/volume specified\n"
        self.display_label.config(text=display_text)


    
    def merge_masks(self, new_mask, existing_mask, new_category_index): 
        if existing_mask is None:
           merged_mask = new_mask.copy() 
        else:
           merged_mask = existing_mask.copy()
           
        new_category_areas = new_mask == 1  
        merged_mask[new_category_areas] = new_category_index
        
        return merged_mask

    def reset_canvas(self):
        self.canvas.delete("all")  
    def is_within_image_bounds(self, x, y):
        return 0 <= x < self.image.width and 0 <= y < self.image.height

    def include_left_click(self, event):
        if self.include_click_count < 10:
            x, y = event.x, event.y
            if self.is_within_image_bounds(x, y):
                self.include_pixels.append((x, y)) 
                self.highlight_pixels(x, y, "#84F8ED", 5)
                self.update_labels()
                self.include_click_count += 1
                self.action_history.append("include")

    def exclude_right_click(self, event):
        if self.exclude_click_count < 10:
            x, y = event.x, event.y
            if self.is_within_image_bounds(x, y):
                self.exclude_pixels.append((x, y))  
                self.highlight_pixels(x, y, "#F792C4", 5)
                self.update_labels()
                self.exclude_click_count += 1
                self.action_history.append("exclude")
                
    def highlight_pixels(self, x, y, color, radius):
        if self.is_within_image_bounds(x, y):
            start_x = x - radius
            start_y = y - radius
            end_x = x + radius
            end_y = y + radius
            self.canvas.create_oval(start_x, start_y, end_x, end_y,
                                    outline=color, fill=color, tags=("highlighted_pixel", f"pixel{len(self.include_pixels) + len(self.exclude_pixels)}"))

    def clear_points(self):
        self.include_pixels = []
        self.exclude_pixels = []
        self.include_click_count = 0
        self.exclude_click_count = 0
        self.update_labels()
        self.canvas.delete("highlighted_pixel")  

    def undo_point(self):
        if self.action_history:
            last_action = self.action_history.pop()
            if last_action == "include" and self.include_pixels:
                self.include_pixels.pop()
                self.include_click_count -= 1
              
                self.canvas.delete(f"pixel{len(self.include_pixels) + len(self.exclude_pixels) + 1}")
            elif last_action == "exclude" and self.exclude_pixels:
                self.exclude_pixels.pop()
                self.exclude_click_count -= 1
              
                self.canvas.delete(f"pixel{len(self.include_pixels) + len(self.exclude_pixels) + 1}")
            self.update_labels()

    def clear_canvas(self, canvas_item):
        self.canvas.delete(canvas_item)

    def redraw_points(self, points, color):
        for x, y in points:
            self.highlight_pixels(x, y, color, 5)


    def update_labels(self):
        self.include_label.config(text=f"Include Pixels: {self.include_pixels}")
        self.exclude_label.config(text=f"Exclude Pixels: {self.exclude_pixels}")

   

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
