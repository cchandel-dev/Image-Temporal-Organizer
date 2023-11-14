import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageAnnotator:
    def __init__(self, root, image_folder, label_folder):
        self.root = root
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_index = 0
        self.rectangles = []  # To store rectangle coordinates
        self.current_rectangle = None  # To store the current drawing rectangle
        # New attributes to store initial mouse click coordinates
        self.start_x = None
        self.start_y = None
        # Create GUI components
        self.canvas = tk.Canvas(root)
        self.canvas.pack()


         # Add a label
        self.label = tk.Label(root, text="Your Label Text")
        self.label.pack(side=tk.TOP)

        # Load the first image
        self.load_image()
        self.current_index += 1
        # Bind keyboard and mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_drawing_rectangle)
        self.canvas.bind("<B1-Motion>", self.drawing_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.finish_drawing_rectangle)


    def load_image(self):
        image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        self.image = cv2.imread(image_path)
        self.display_image()
        self.label.destroy()
        self.label = tk.Label(root, text= self.image_list[self.current_index])
        self.label.pack(side=tk.TOP)

    def display_image(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_tk = Image.fromarray(image_rgb)

        # Resize the image to a smaller size (e.g., 50%)
        new_width = int(image_tk.width * 0.5)
        new_height = int(image_tk.height * 0.5)
        image_tk = image_tk.resize((new_width, new_height), 0)

        self.photo = ImageTk.PhotoImage(image=image_tk)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        # Send the canvas (image) to the back
        self.canvas.lower("all")


    def prev_image(self, event):
        self.current_index = max(0, self.current_index - 1)
        self.load_image()
        self.clear_rectangles()

    def next_image(self):
        self.save_labels()
        if  self.current_index < len(self.image_list) - 1:
            self.load_image()
            self.clear_rectangles()
        else:
            root.destroy()
        self.current_index += 1

    def start_drawing_rectangle(self, event):
        self.start_x, self.start_y = event.x, event.y
        print("starting_x: ", event.x, "starting_y: ", event.y)

    def drawing_rectangle(self, event):
        self.canvas.delete(self.current_rectangle)
        self.current_rectangle = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y)

    def finish_drawing_rectangle(self, event):
        print("ending_x: ", event.x, "ending_y: ", event.y)
        min_x = min(self.start_x, event.x)
        max_x = max(self.start_x, event.x)
        min_y = min(self.start_y, event.y)
        max_y = max(self.start_y, event.y)
        self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y)
        self.rectangles.append([min_x, min_y, max_x - min_x, max_y - min_y])
        print("rectangle_x", min_x, "rectangle_y", min_y, "width", max_x - min_x, "height", max_y - min_y )
        image_width, image_height = self.image.shape[1] * 0.5, self.image.shape[0] *0.5
        print("normalized x", min_x/image_width, "normalized y", min_y/image_height, "normalized width", (max_x - min_x)/image_width,"normalized height", (max_y - min_y)/image_height )

    def save_labels(self):
        # Save labels in YOLO format
        image_width, image_height = self.image.shape[1] * 0.5, self.image.shape[0] * 0.5
        
        print("labels image width: ", image_width, "labels image height: ", image_height)
        with open(os.path.join(label_folder, self.image_list[self.current_index - 1][:-4]+".txt"), 'w') as file:
            for rect_coords in self.rectangles:
                x, y, w, h = rect_coords

                # Normalize coordinates
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                width = w / image_width
                height = h / image_height
                print("normalized x center", x_center, "normalized y center", y_center, "normalized width", width, "normalized height", height )
                label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} \n"
                file.write(label)
        self.rectangles = []

    def clear_rectangles(self):
        self.rectangles = []
        self.canvas.delete("all")
        self.display_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Annotator")

    # Ask user to select a folder containing images
    image_folder = filedialog.askdirectory(title="Select Image Folder")
    label_folder = filedialog.askdirectory(title="Select Empty Label Folder")

    if image_folder:
        annotator = ImageAnnotator(root, image_folder, label_folder)
        


        # Additional buttons for clearing and saving annotations
        clear_button = tk.Button(root, text="Clear Rectangles", command=annotator.clear_rectangles)
        clear_button.pack(side=tk.TOP)

        save_button = tk.Button(root, text="Next Image", command=annotator.next_image)
        save_button.pack(side=tk.BOTTOM)

        root.mainloop()
