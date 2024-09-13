# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:25:01 2024

@author: lubna
"""

import tkinter as tk
from tkinter import filedialog, Canvas, Radiobutton, IntVar, Scale, HORIZONTAL, StringVar, OptionMenu
from PIL import Image, ImageTk, ImageDraw, ImageOps

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

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

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack()

        self.overlay_canvas = Canvas(self.canvas_frame, width=300, height=300)
        self.overlay_canvas.grid(row=0, column=0)
        self.mask_canvas = Canvas(self.canvas_frame, width=300, height=300)
        self.mask_canvas.grid(row=0, column=1)

        self.overlay_canvas.bind("<B1-Motion>", self.paint)
        self.overlay_canvas.bind("<ButtonRelease-1>", self.update_mask)
        self.overlay_canvas.bind("<Button-1>", self.paint)
        self.overlay_canvas.bind("<Motion>", self.preview_paint)

        self.image = None
        self.mask = None
        self.tk_mask = None
        self.tk_overlay = None
        self.preview_overlay = None

    def toggle_brush_options(self):
        if self.brush_active.get() == 1:
            self.brush_action_dropdown.config(state=tk.NORMAL)
        else:
            self.brush_action_dropdown.config(state=tk.DISABLED)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            self.image = ImageOps.fit(image, (300, 300))
            self.tk_image = ImageTk.PhotoImage(self.image)

            self.mask = Image.new('1', (300, 300), 0)
            self.mask_draw = ImageDraw.Draw(self.mask)
            self.tk_mask = ImageTk.PhotoImage(self.mask.convert("L"))
            self.mask_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_mask)

            self.update_overlay()

    def paint(self, event):
        if self.brush_active.get() == 1 and self.image:
            x, y = event.x, event.y
            brush_size = self.brush_size_slider.get()
            action = self.brush_action.get()
            if action == "Add":
                self.mask_draw.ellipse([x-brush_size, y-brush_size, x+brush_size, y+brush_size], fill=1)
            elif action == "Delete":
                self.mask_draw.ellipse([x-brush_size, y-brush_size, x+brush_size, y+brush_size], fill=0)
            self.update_mask()
            self.update_overlay()

    def update_mask(self, event=None):
        self.tk_mask = ImageTk.PhotoImage(self.mask.convert("L"))
        self.mask_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_mask)

    def update_overlay(self):
        if self.image and self.mask:
            overlay = self.image.copy().convert("RGBA")
            mask_rgba = Image.new("RGBA", self.mask.size)
            mask_rgba.paste((0, 255, 0, 128), mask=self.mask.convert("L"))  
            overlay = Image.alpha_composite(overlay, mask_rgba)
            self.tk_overlay = ImageTk.PhotoImage(overlay.convert("RGB"))
            self.overlay_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_overlay)

    def preview_paint(self, event):
        if self.brush_active.get() == 1 and self.image:
            x, y = event.x, event.y
            brush_size = self.brush_size_slider.get()
            overlay = self.image.copy().convert("RGBA")
            mask_rgba = Image.new("RGBA", self.mask.size)
            mask_rgba.paste((0, 255, 0, 128), mask=self.mask.convert("L"))  
            action = self.brush_action.get()
            draw = ImageDraw.Draw(mask_rgba)
            if action == "Add":
                draw.ellipse([x-brush_size, y-brush_size, x+brush_size, y+brush_size], fill=(0, 255, 0, 128))
            elif action == "Delete":
                draw.ellipse([x-brush_size, y-brush_size, x+brush_size, y+brush_size], outline=(255, 0, 0, 255), width=3)
            overlay = Image.alpha_composite(overlay, mask_rgba)
            self.preview_overlay = ImageTk.PhotoImage(overlay.convert("RGB"))
            self.overlay_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_overlay)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()