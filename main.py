# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:16:30 2024

@author: Lubnaa Abdur Rahman


Copyright ©  2024 University of Bern, ARTORG Center for Biomedical Engineering Research, [Lubnaa Abdur Rahman, Ioannis Papathanail, Lorenzo Brigato, Stavroula Mougiakakou]

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
from tkinter import filedialog, Menu, Radiobutton, IntVar, Scale, HORIZONTAL, StringVar, OptionMenu
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import torch
import json
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from util import build_sam_vit_b


def resource_path(relative_path):
    import sys
    import os
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def update_sam_model(model_type, sam_checkpoint):
    global sam  # Indicate that we're updating the global variable
    device = "cpu" if model_type in ["vit_l", "vit_h"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)


sam_model_registry = {
    "vit_b": build_sam_vit_b,
    "MealSAM": build_sam_vit_b,
}


class ScrolledListbox(tk.Toplevel):
    def __init__(self, parent, options, var, icon=None, close_callback=None, select_callback=None, previous_focus=None, **kwargs):
        super().__init__(parent)
        self.bind("<FocusOut>", self.on_focus_out)
        self.previous_focus = previous_focus
        self.overrideredirect(True)
        self.transient(parent)
        self.var = var
        self.close_callback = close_callback
        self.select_callback = select_callback
        if icon:
            self.iconphoto(False, icon)

        self.listbox = tk.Listbox(self, **kwargs)
        self.listbox.config(height=10, width=60)  # Add this line
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=self.scrollbar.set)

        for option in options:
            self.listbox.insert(tk.END, option)

        self.listbox.selection_set(0)  # Preselect the first item
        self.listbox.activate(0)

        self.listbox.bind("<ButtonRelease-1>", self.selected)
        self.listbox.bind("<Return>", self.selected)

    def selected(self, event):
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            value = self.listbox.get(index)
            self.var.set(value)
            if self.select_callback:
                self.select_callback()
            if self.close_callback:
                self.close_callback()

    def on_focus_out(self, event):
        if self.close_callback:
            self.close_callback()


class AutocompleteCombobox(ttk.Entry):
    def __init__(self, parent, categories, app_instance, **kwargs):
        super().__init__(parent, **kwargs)
        self.config(width=60)  # Increase number of characters visible
        self.app_instance = app_instance
        self.categories = categories
        self.dropdown = None
        self.var = self["textvariable"] = kwargs.get("textvariable", tk.StringVar())
        self.var.trace_add("write", self.update_suggestions)

        self.bind("<KeyRelease>", self.on_keyrelease)
        # self.bind("<Down>", self.show_dropdown)
        self.bind("<Return>", self.check_add_new_category)
        self.bind("<Down>", self.move_selection_down)
        self.bind("<Up>", self.move_selection_up)

    def on_keyrelease(self, event):
        if event.keysym == "Escape":
            self.close_dropdown()
            return

    def show_dropdown(self, event=None):
        value = self.var.get().strip()
        if value:
            matches = [c for c in self.categories if value.lower() in c.lower()]
            if "Add new category..." not in matches:
                matches.append("Add new category...")
        else:
            matches = self.categories + ["Add new category..."]

        if matches:
            self.close_dropdown()
            x = self.winfo_rootx()
            y = self.winfo_rooty() + self.winfo_height()
            self.dropdown = ScrolledListbox(self.winfo_toplevel(),
                                            matches,
                                            self.var,
                                            icon=self.app_instance.icon_image,
                                            close_callback=self.close_dropdown,
                                            select_callback=self.on_select_from_list,
                                            previous_focus=self.previous_focus)
            self.dropdown.geometry(f"+{x}+{y}")
            self.dropdown.deiconify()

    def update_suggestions(self, *args):
        if not self.dropdown:
            self.previous_focus = self.focus_get()
        self.show_dropdown()

    def close_dropdown(self):
        if self.dropdown:
            try:
                self.dropdown.grab_release()
            except tk.TclError:
                pass
            self.dropdown.destroy()
            self.dropdown = None

    def check_add_new_category(self, event=None):
        if self.dropdown:
            # Get selected item from the listbox
            selection = self.dropdown.listbox.curselection()
            if selection:
                index = selection[0]
                value = self.dropdown.listbox.get(index)
                self.var.set(value)
            self.close_dropdown()

        elif self.var.get() == "Add new category...":
            self.app_instance.add_new_category()
            self.close_dropdown()

    def move_selection_down(self, event=None):
        if self.dropdown and self.dropdown.listbox.size() > 0:
            current = self.dropdown.listbox.curselection()
            if current:
                index = current[0]
                if index < self.dropdown.listbox.size() - 1:
                    self.dropdown.listbox.selection_clear(0, tk.END)
                    self.dropdown.listbox.selection_set(index + 1)
                    self.dropdown.listbox.activate(index + 1)
            else:
                self.dropdown.listbox.selection_set(0)
                self.dropdown.listbox.activate(0)
        else:
            self.show_dropdown()

    def move_selection_up(self, event=None):
        if self.dropdown and self.dropdown.listbox.size() > 0:
            current = self.dropdown.listbox.curselection()
            if current:
                index = current[0]
                if index > 0:
                    self.dropdown.listbox.selection_clear(0, tk.END)
                    self.dropdown.listbox.selection_set(index - 1)
                    self.dropdown.listbox.activate(index - 1)

    def on_select_from_list(self):
        self.winfo_toplevel().focus_force()

        def restore_focus():
            try:
                self.previous_focus.focus_set()
                self.previous_focus.event_generate("<FocusIn>")
                self.previous_focus.update()  # Force widget update
                self.previous_focus.lift()    # Bring it to the front
            except tk.TclError:
                pass

        # Let the dropdown destroy first, then restore focus
        self.after(10, restore_focus)


class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.model_type = "MealSAM"  # Default model type
        self.sam_checkpoint = resource_path("weights/MealSAM.pth")
        self.action_history = []
        self.semi_segmented_mask = None

        self.model_variable = tk.StringVar(value="MealSAM")
        self.mask_generator = self.update_model_selection()

        model_options = ["MealSAM"]

        self.root.title("Food Annotator")
        icon_path = resource_path("tool_resources/appicon.png")
        icon = ImageTk.PhotoImage(file=icon_path)
        self.icon_image = icon
        self.root.iconphoto(False, icon)

        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        self.image_uploaded = False

        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Upload", command=self.upload_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)

        up_img_path = resource_path("tool_resources/upload.png")
        upload_image_icon = ImageTk.PhotoImage(Image.open(up_img_path).resize((20, 20)))
        sv_img_path = resource_path("tool_resources/save.png")
        save_image_icon = ImageTk.PhotoImage(Image.open(sv_img_path).resize((20, 20)))

        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", anchor="n", pady=5)

        # Upload Image
        self.upload_button = tk.Button(button_frame, text="Upload", image=upload_image_icon, compound="left",
                                       command=self.upload_image)
        self.upload_button.image = upload_image_icon
        self.upload_button.pack(side="left", padx=5)

        # Save Files - Save Validated Mask, and resized image
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

        # Clears the include and exclude points
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_points)
        self.clear_button.pack(side="left", padx=5)

        # Undo previously clicked point on the RGB
        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo_point)
        self.undo_button.pack(side="left", padx=5)

        # Provided include/exclude points Mono Mask generated for object
        self.semi_segment_button = tk.Button(button_frame, text="Segment", command=self.semi_segment)
        self.semi_segment_button.pack(side="left", padx=5)

        self.brush_active = IntVar(value=0)
        self.brush_radio = Radiobutton(root, text="Activate Brush", variable=self.brush_active, value=1, command=self.toggle_brush_options)
        self.brush_radio.pack()

        self.deactivate_brush_radio = Radiobutton(root, text="Deactivate Brush", variable=self.brush_active, value=0, command=self.toggle_brush_options)
        self.deactivate_brush_radio.pack()

        self.brush_action = StringVar(value="Add")
        self.brush_action_dropdown = OptionMenu(root, self.brush_action, "Add", "Delete")
        self.brush_action_dropdown.pack()
        self.brush_action_dropdown.config(state=tk.DISABLED)

        self.brush_size_slider = Scale(root, from_=1, to=50, orient=HORIZONTAL, label="Brush Size")
        self.brush_size_slider.pack()
        self.brush_size_slider.set(10)

        self.brush_size = self.brush_size_slider.get()
        self.brush_id = None
        self.brush_image = None
        self.paint_active = False
        self.colors = []
        self.current_label = 1  # Initialize current_label
        # Food categories/labels to assign per specific segment
        self.category_variable = tk.StringVar(root)
        self.load_categories_from_json(resource_path("tool_resources/categories.json"))
        self.category_dropdown_label = tk.Label(button_frame, text="    Category:")
        self.category_dropdown_label.pack(side="left", padx=5)
        self.category_dropdown = AutocompleteCombobox(button_frame, self.categories, self, textvariable=self.category_variable)
        self.category_dropdown.pack(side="left", padx=5)
        style = ttk.Style(self.root)
        style.theme_use("alt")
        style.configure("TMenubutton", background="white", foreground="black")
        style.map("TMenubutton", background=[("active", "grey")])

        self.annotation_option = tk.StringVar(value="No")
        self.annotation_type = tk.StringVar(value="Weight")

        self.annotation_option_label = tk.Label(button_frame, text="Others (vol/weight/CHO)?")
        self.annotation_option_label.pack(side="left", padx=5)
        self.yes_radio_button = tk.Radiobutton(button_frame, text="Yes", variable=self.annotation_option, value="Yes", command=self.toggle_annotation_fields)
        self.no_radio_button = tk.Radiobutton(button_frame, text="No", variable=self.annotation_option, value="No", command=self.toggle_annotation_fields)
        self.yes_radio_button.pack(side="left", padx=5)
        self.no_radio_button.pack(side="left", padx=5)

        self.annotation_type_label = tk.Label(button_frame, text="Type:", state="disabled")
        self.annotation_type_menu = ttk.Combobox(button_frame, textvariable=self.annotation_type, state="disabled", values=["Weight", "Volume", "Portion", "Carbohydrates"])
        self.annotation_type_label.pack(side="left", padx=5)
        self.annotation_type_menu.pack(side="left", padx=5)

        self.grams_label = tk.Label(button_frame, text="Quantity:", state="disabled")
        self.grams_label.pack(side="left", padx=10)

        self.grams_entry = tk.Entry(button_frame, state="disabled")
        self.grams_entry.pack(side="left", padx=10)

        self.yes_radio_button = tk.Radiobutton(button_frame, text="Yes", variable=self.annotation_option, value="Yes", command=self.toggle_annotation_fields)
        self.no_radio_button = tk.Radiobutton(button_frame, text="No", variable=self.annotation_option, value="No", command=self.toggle_annotation_fields)

        self.validate_mask_button = tk.Button(button_frame, text="Validate", command=self.validate_mask)
        self.validate_mask_button.pack(side="left", padx=10)

        self.image_on_canvas = None
        self.image_path = None
        self.photo_image = None
        self.semi_segmented_mask = None
        self.mask_image_on_canvas = None
        self.mask_photo_image = None
        self.overlaid_mask_canvas = None
        self.overlaid_validated_mask_canvas = None
        self.include_pixels = []
        self.exclude_pixels = []
        self.include_click_count = 0
        self.exclude_click_count = 0
        self.image_directory = ""
        self.display_label = []
        self.segment_data = {}
        self.all_nutrient_data = []

        # Include/Exclude points
        footer_text = (
            "Copyright © 2024 University of Bern, ARTORG Center for Biomedical Engineering Research, "
            "[Lubnaa Abdur Rahman, Ioannis Papathanail, Lorenzo Brigato, Stavroula Mougiakakou]"
        )
        self.footer_label = tk.Label(self.root, text=footer_text, anchor="center", justify="center", wraplength=self.root.winfo_screenwidth())
        self.footer_label.pack(side="bottom", fill="x", pady=(10, 5))

        self.include_label = tk.Label(self.root, text="Include Pixels: ")
        self.include_label.pack(side="bottom")
        self.exclude_label = tk.Label(self.root, text="Exclude Pixels: ")
        self.exclude_label.pack(side="bottom")

        self.display_label = tk.Label(self.root, text="Category and weight: ")
        self.display_label.pack(side="bottom")

        # Clears all the canvas items except the RGB image
        self.clear_all_button = tk.Button(button_frame, text="Clear All", command=self.clear_canvas_all)
        self.clear_all_button.pack(side="left", padx=5)

        # RGB Image
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True, side="left")
        self.rgb_image_label = tk.Label(self.canvas_frame, text="RGB Image")
        self.rgb_image_label.pack()
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        # Colored mask generated by SAM either by clicking "Segment" or "Semi-segment" buttons
        self.mask_canvas_frame = tk.Frame(self.root)
        self.mask_canvas_frame.pack(fill="both", expand=True, side="left")
        self.mask_label = tk.Label(self.mask_canvas_frame, text="Model's Mask")
        self.mask_label.pack()
        self.mask_canvas = tk.Canvas(self.mask_canvas_frame, cursor="cross")
        self.mask_canvas.pack(fill="both", expand=True)

        # Overlaid mask generated by SAM on top of the RGB
        self.overlaid_mask_frame = tk.Frame(self.root)
        self.overlaid_mask_frame.pack(fill="both", expand=True, side="left")
        self.overlaid_mask_label = tk.Label(self.overlaid_mask_frame, text="Overlaid Model's mask")
        self.overlaid_mask_label.pack()
        self.overlaid_mask_canvas = tk.Canvas(self.overlaid_mask_frame, cursor="cross")
        self.overlaid_mask_canvas.pack(fill="both", expand=True)

        self.overlaid_mask_canvas.bind("<Motion>", self.draw_brush)  # Allow brush drawing on top of the overlaid mask
        # Colored validated mask - can contain more than one segmented object - Validate button must be clicked
        self.validated_mask_frame = tk.Frame(self.root)
        self.validated_mask_frame.pack(fill="both", expand=True, side="left")
        self.validated_mask_label = tk.Label(self.validated_mask_frame, text="Validated Mask")
        self.validated_mask_label.pack()
        self.validated_mask_canvas = tk.Canvas(self.validated_mask_frame, cursor="cross")
        self.validated_mask_canvas.pack(fill="both", expand=True)

        # Overlaid validated mask on the RGB
        self.overlaid_validated_mask_frame = tk.Frame(self.root)
        self.overlaid_validated_mask_frame.pack(fill="both", expand=True, side="right")
        self.overlaid_validated_mask_label = tk.Label(self.overlaid_validated_mask_frame, text="Overlaid Validated Mask")
        self.overlaid_validated_mask_label.pack()
        self.overlaid_validated_mask_canvas = tk.Canvas(self.overlaid_validated_mask_frame, cursor="cross")
        self.overlaid_validated_mask_canvas.pack(fill="both", expand=True)

    def create_brush_image(self, size, color, alpha):
        brush_image = Image.new("RGBA", (size * 2, size * 2), (255, 255, 255, 0))
        draw = ImageDraw.Draw(brush_image)
        draw.ellipse((0, 0, size * 2, size * 2), fill=color + (int(255 * alpha),))
        return brush_image

    def draw_brush(self, event):
        if self.brush_active.get() == 1:
            x, y = event.x, event.y
            size = self.brush_size_slider.get()
            action = self.brush_action.get()

            brush_color = (255, 0, 0) if action == 'Delete' else (0, 255, 0)
            alpha = 0.5

            self.brush_image = self.create_brush_image(size, brush_color, alpha)
            self.tk_brush_image = ImageTk.PhotoImage(self.brush_image)

            if self.brush_id:
                self.overlaid_mask_canvas.delete(self.brush_id)
            self.brush_id = self.overlaid_mask_canvas.create_image(x, y, image=self.tk_brush_image, anchor="center")
            self.overlaid_mask_canvas.lift(self.brush_id)

            if self.paint_active:
                self.paint(event)

    def clear_brush(self, event):
        if self.brush_id:
            self.overlaid_mask_canvas.delete(self.brush_id)
            self.brush_id = None

    def start_paint(self, event):
        self.paint_active = True
        x, y = event.x, event.y
        if 0 <= x < self.merged_mask.shape[1] and 0 <= y < self.merged_mask.shape[0]:
            self.current_label = self.merged_mask[y, x]
            if self.current_label == 0 and self.brush_action.get() == 'Add':

                self.max_label += 1
                self.current_label = self.max_label
                self.colors.append(np.random.randint(0, 256, 3).tolist())
        else:
            self.current_label = 1
        self.paint(event)

    def stop_paint(self, event):
        self.paint_active = False
        self.clear_brush(event)
        # Update the mask on the canvas
        self.update_model_mask_on_canvas()
        self.update_overlaid_mask_on_canvas()

    def apply_brush_to_mask(self, x, y, size, action):
        if action == 'Add':
            new_label = self.current_label
            if new_label == 0:

                self.max_label += 1
                new_label = self.max_label
                self.colors.append(np.random.randint(0, 256, 3).tolist())
                self.current_label = new_label
        elif action == 'Delete':
            new_label = 0

        for i in range(-size, size):
            for j in range(-size, size):
                if i**2 + j**2 <= size**2:
                    x_i = x + i
                    y_j = y + j
                    if 0 <= x_i < self.merged_mask.shape[1] and 0 <= y_j < self.merged_mask.shape[0]:
                        self.merged_mask[y_j, x_i] = new_label

    def update_model_mask_on_canvas(self):

        color_mask = self.colorize_mask(self.merged_mask)
        mask_image = Image.fromarray(color_mask)
        self.tk_model_mask_image = ImageTk.PhotoImage(mask_image)
        self.mask_canvas.delete("all")
        self.mask_canvas.create_image(0, 0, anchor="nw", image=self.tk_model_mask_image)
        self.mask_canvas.image = self.tk_model_mask_image

    def update_overlaid_mask_on_canvas(self):

        color_mask = self.colorize_mask(self.merged_mask)

        overlaid_image_array = np.array(self.image.convert("RGB"))
        alpha = 0.5
        mask_indices = np.any(color_mask != [0, 0, 0], axis=-1)
        overlaid_image_array[mask_indices] = (
            alpha * color_mask[mask_indices] + (1 - alpha) * overlaid_image_array[mask_indices]
        ).astype("uint8")

        overlaid_mask_image = Image.fromarray(overlaid_image_array)
        self.tk_overlaid_mask_image = ImageTk.PhotoImage(overlaid_mask_image)
        self.overlaid_mask_canvas.delete("all")
        self.overlaid_mask_canvas.create_image(0, 0, anchor="nw", image=self.tk_overlaid_mask_image)
        self.overlaid_mask_canvas.image = self.tk_overlaid_mask_image

    def colorize_mask(self, mask):
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:
                continue
            if label > len(self.colors):
                self.colors.append(np.random.randint(0, 256, 3).tolist())
            color = np.array(self.colors[label - 1])
            color_mask[mask == label] = color
        return color_mask

    def paint(self, event):
        x, y = event.x, event.y
        size = self.brush_size_slider.get()
        action = self.brush_action.get()
        self.apply_brush_to_mask(x, y, size, action)
        self.update_model_mask_on_canvas()
        self.update_overlaid_mask_on_canvas()

    def paint_once(self, event):
        self.paint_active = True
        self.paint(event)
        self.paint_active = False

    def toggle_brush_options(self):
        if self.brush_active.get() == 1:
            if self.mask_photo_image is None:
                self.create_mask()

            self.brush_action_dropdown.config(state=tk.NORMAL)
            self.brush_size_slider.config(state=tk.NORMAL)
            self.semi_segment_button.config(state=tk.DISABLED)
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")
            self.overlaid_mask_canvas.bind("<Motion>", self.draw_brush)
            self.overlaid_mask_canvas.bind("<Leave>", self.clear_brush)

            self.overlaid_mask_canvas.bind("<ButtonPress-1>", self.start_paint)
            self.overlaid_mask_canvas.bind("<ButtonRelease-1>", self.stop_paint)
            self.overlaid_mask_canvas.bind("<B1-Motion>", self.paint)
            self.overlaid_mask_canvas.bind("<Button-1>", self.paint_once)

        else:
            self.brush_action_dropdown.config(state=tk.DISABLED)
            self.brush_size_slider.config(state=tk.DISABLED)
            self.semi_segment_button.config(state=tk.NORMAL)
            self.canvas.bind("<Button-1>", self.include_left_click)
            self.canvas.bind("<Button-3>", self.exclude_right_click)

            self.overlaid_mask_canvas.unbind("<Motion>")
            self.overlaid_mask_canvas.unbind("<Leave>")

            self.overlaid_mask_canvas.unbind("<ButtonPress-1>")
            self.overlaid_mask_canvas.unbind("<ButtonRelease-1>")
            self.overlaid_mask_canvas.unbind("<B1-Motion>")
            self.overlaid_mask_canvas.unbind("<Button-1>")

            self.clear_brush(None)

    def determine_checkpoint_path(self, model_type):
        if model_type == "MealSAM":
            return resource_path("weights/MealSAM.pth")

    def update_model_selection(self, event=None):
        self.model_type = self.model_variable.get()
        if self.model_type == "MealSAM":
            self.model_type = "vit_b"
            self.sam_checkpoint = resource_path("weights/MealSAM.pth")

        # Initialize SAM model with current selection
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        device = "cpu" if self.model_type in ["vit_l", "vit_h"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def add_new_category(self):
        new_category_name = tk.simpledialog.askstring("New Category", "Enter new category name:")
        if new_category_name and new_category_name.strip() and new_category_name not in self.categories:
            self.categories.append(new_category_name)
            self.category_dropdown["values"] = self.categories + ["Add new category..."]
            self.update_categories_json()
            self.category_variable.set(new_category_name)

    def update_categories_json(self):
        #
        try:
            with open(resource_path("tool_resources/categories.json"), "w") as file:
                json.dump(self.categories, file)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to update categories: {e}")

    def toggle_annotation_fields(self):
        self.grams_entry.delete(0, tk.END)
        if self.annotation_option.get() == "Yes":
            #  annotation type selection (weight/volume)
            self.annotation_type_label.config(state="normal")
            self.annotation_type_menu.config(state="readonly")
            self.grams_label.config(state="normal")
            self.grams_entry.config(state="normal")
            self.annotation_type.set("Weight")
            self.grams_entry.focus_set()
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

    # Logic for Upload button
    def upload_image(self):
        self.include_pixels = []
        self.exclude_pixels = []
        self.all_nutrient_data = []
        self.update_nutrient_data_display()

        self.include_click_count = 0
        self.exclude_click_count = 0

        self.include_label.config(text="Include Pixels (x,y): ")  # Include pixels are assigned a label of 1 - i.e. Foreground -- appear blue
        self.exclude_label.config(text="Exclude Pixels (x,y): ")  # Exclude pixels are assigned a label of 0 - i.e. Background -- appear pink

        if hasattr(self, "validated_mask"):
            del self.validated_mask

        if hasattr(self, "val_copy"):
            del self.val_copy

        filetypes = [("JPG files", "*.jpg"), ("PNG files", "*.png")]
        self.image_path = filedialog.askopenfilename(filetypes=filetypes)
        if not self.image_path:
            return

        # Save the directory for proper file saving later in same directory as previously
        self.image_directory = os.path.dirname(self.image_path)
        self.image_filename = os.path.basename(self.image_path)

        self.image_uploaded = True
        original_image = Image.open(self.image_path)

        if original_image.width != 380:
            new_height = int(original_image.height * (380 / original_image.width))
            original_image = original_image.resize((380, new_height))

        self.image = original_image
        self.merged_mask = None
        self.merged_mask = np.zeros((self.image.height, self.image.width), dtype=np.int32)
        self.max_label = 0
        self.photo_image = ImageTk.PhotoImage(self.image)

        self.overlaid_mask_canvas.delete("all")
        self.validated_mask_canvas.delete("all")
        self.overlaid_validated_mask_canvas.delete("all")

        self.create_mask()
        self.reset_canvas()
        self.mask_canvas.delete("all")

        self.overlaid_mask_canvas.delete("all")
        self.validated_mask_canvas.delete("all")
        self.overlaid_validated_mask_canvas.delete("all")

        self.create_mask()
        self.reset_canvas()
        self.mask_canvas.delete("all")

        self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.image = self.photo_image

        self.canvas.bind("<Button-1>", self.include_left_click)
        self.canvas.bind("<Button-3>", self.exclude_right_click)

        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor="nw", image=self.photo_image)

        self.canvas.image = self.photo_image

        self.canvas.bind("<Button-1>", self.include_left_click)
        self.canvas.bind("<Button-3>", self.exclude_right_click)

        self.overlaid_mask_canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.overlaid_mask_canvas.image = self.photo_image

    def save_image(self):
        if hasattr(self, "image_filename"):
            original_filename, original_extension = os.path.splitext(self.image_filename)

        if self.image_path:
            if hasattr(self, "val_copy"):

                validated_mask_image = self.val_copy
                val_filename = f"{original_filename}_validated_mask.png"
                val_path = os.path.join(self.image_directory, val_filename)
                cv2.imwrite(val_path, validated_mask_image.astype(np.uint16))
                output_filename = f"{original_filename}_resized.png"
                output_path = os.path.join(self.image_directory, output_filename)
                self.image.save(output_path)

                annotated_weight_filename = f"{original_filename}_weights_info.txt"
                nutrient_info_path = os.path.join(self.image_directory, annotated_weight_filename)
                with open(nutrient_info_path, 'w') as nutrient_file:
                    for entry in self.all_nutrient_data:
                        weight = entry.get('Weight')
                        volume = entry.get('Volume')
                        portions = entry.get('Portion')
                        carbohydrates = entry.get('Carbohydrates')
                        if weight:
                            nutrient_file.write(f"{entry['Category']}: {weight} grams\n")
                        elif volume:
                            nutrient_file.write(f"{entry['Category']}: {volume} ml\n")
                        elif portions:
                            nutrient_file.write(f"{entry['Category']}: {portions} portions\n")
                        elif carbohydrates:
                            nutrient_file.write(f"{entry['Category']}: {carbohydrates} grams CHO\n")
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

    # Used by clear all button
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
        self.category_dropdown.close_dropdown()
        self.validate_mask = None
        self.validated_mask = None

    # Unique color for each single category
    def create_color_map(self):
        num_categories = len(self.categories)
        num_categories += 1
        color_map = np.zeros((num_categories, 3), dtype=np.uint16)
        for i in range(num_categories):
            color_map[i] = [i * 20 % 256, i * 30 % 256, i * 50 % 256]
        return color_map

    # Semi-segment button logic - considers include/exclude points
    def semi_segment(self):
        if not self.image_uploaded:
            tk.messagebox.showerror("Error", "Please upload an image first.")
            return

        if not self.include_pixels:
            tk.messagebox.showerror("Error", "Semi Segment needs include points")
            return

        if self.exclude_pixels:
            include_coords = np.asarray(self.include_pixels)
            exclude_coords = np.asarray(self.exclude_pixels)
            include_labels = np.array([1] * len(self.include_pixels))  # Include is foreground
            exclude_labels = np.array([0] * len(self.exclude_pixels))  # Exclude is background
            inputarray = np.concatenate((include_coords, exclude_coords))
            input_label = np.concatenate((include_labels, exclude_labels))
        else:
            inputarray = np.asarray(self.include_pixels)
            input_label = np.array([1] * len(self.include_pixels))

        # Call model - multimasks not to be generated
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        device = "cpu" if self.model_type in ["vit_l", "vit_h"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.colors = []  # Reset colors
        for i in range(num_masks):
            mask_value = i + 1
            merged_mask[masks[i]] = mask_value
            random_color = np.random.randint(0, 256, size=3).tolist()
            self.colors.append(random_color)

        self.merged_mask = merged_mask
        self.max_label = np.max(self.merged_mask)

        color_mask = self.colorize_mask(self.merged_mask)

        mask_indices = np.any(color_mask != [0, 0, 0], axis=-1)
        overlay_image_array = image_array.copy()
        overlay_image_array = cv2.cvtColor(overlay_image_array, cv2.COLOR_BGR2RGB)
        alpha = 0.5
        overlay_image_array[mask_indices] = (
            alpha * color_mask[mask_indices] + (1 - alpha) * image_array[mask_indices]
        ).astype("uint8")

        self.semi_segmented_mask = color_mask

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
            if self.merged_mask is not None:
                self.semi_segmented_mask = self.colorize_mask(self.merged_mask)
                tk.messagebox.showinfo("Info", "No semi-segmented mask found. Using existing mask instead.")
            else:
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

        if self.include_pixels and self.exclude_pixels:
            existing_entry = next((entry for entry in self.all_nutrient_data
                                   if entry['Include Pixels'] == self.include_pixels and entry['Exclude Pixels'] == self.exclude_pixels), None)
        else:
            existing_entry = None

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

        # Merge masks with new category index
        if hasattr(self, "validated_mask"):
            existing_validated_mask = self.validated_mask
        else:
            existing_validated_mask = None

        validated_mask = self.merge_masks(self.merged_mask, existing_validated_mask, category_index)
        self.validated_mask = validated_mask

        self.val_copy = validated_mask.copy()
        validated_mask = validated_mask.astype(np.uint16)  # 11ct
        validated_mask_image = Image.fromarray(validated_mask)
        validated_mask_photo_image = ImageTk.PhotoImage(validated_mask_image)

        self.validated_mask_canvas.delete("all")
        self.validated_mask_canvas.create_image(0, 0, anchor="nw", image=validated_mask_photo_image)
        self.validated_mask_canvas.image = validated_mask_photo_image

        self.display_colored_validated_mask(validated_mask, self.color_map)
        self.display_colored_validated_mask2(validated_mask, self.color_map)

        self.update_nutrient_data_display()

        self.clear_points()
        self.reset_annotation_fields()
        # self.create_mask()
        self.mask_canvas.delete("all")
        self.merged_mask = np.zeros((self.image.height, self.image.width), dtype=np.int32)
        self.create_mask()
        self.overlaid_mask_canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.overlaid_mask_canvas.image = self.photo_image

        # self.reset_canvas()

    def reset_annotation_fields(self):
        # Reset the "Others?" radio button and related UI
        self.annotation_option.set("No")
        self.toggle_annotation_fields()  # This disables the type menu and label

        # Reset the annotation type to default
        self.annotation_type.set("Weight")

        # Clear the quantity input
        self.grams_entry.delete(0, tk.END)

        # Reset the category dropdown
        self.category_variable.set("")  # or set to a default
        self.category_dropdown.close_dropdown()

    def update_nutrient_data_display(self):
        display_text = "Nutrient Data:\n"
        for entry in self.all_nutrient_data:
            display_text += f"{entry['Category']}: "
            if 'Weight' in entry:
                display_text += f"{entry['Weight']} grams\n"
            elif 'Volume' in entry:
                display_text += f"{entry['Volume']} ml\n"
            elif 'Portion' in entry:
                display_text += f"{entry['Portion']} portions\n"
            elif 'Carbohydrates' in entry:
                display_text += f"{entry['Carbohydrates']} grams CHO\n"
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
