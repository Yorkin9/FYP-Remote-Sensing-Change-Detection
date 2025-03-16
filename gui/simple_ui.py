import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as T
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utilities.data_loader import ChangeDetectionDataset
from models.change_detection import ChangeDetectionModel
from models.modeling.common import compute_metrics
from utilities.visualization import plot_comparison

class SimpleChangeDetectionUI:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Remote Sensing Change Detection")

        # Change detection models
        self.model = model
        self.model.eval()

        # Store T1, T2 image paths
        self.image_path_t1 = None
        self.image_path_t2 = None

        # Display T1 and T2  on the screen
        self.img_tk_t1 = None
        self.img_tk_t2 = None

        self.display_size = (200, 200)

        # 1) Create a Frame with T1 and T2 buttons inside
        self.frame_buttons = tk.Frame(root)
        self.frame_buttons.pack(pady=5)

        self.btn_load_t1 = tk.Button(self.frame_buttons, text="Load T1", command=self.load_image_t1)
        self.btn_load_t1.pack(side="left", padx=5)

        self.btn_load_t2 = tk.Button(self.frame_buttons, text="Load T2", command=self.load_image_t2)
        self.btn_load_t2.pack(side="left", padx=5)

        # Display T1 and T2
        self.frame_images = tk.Frame(root)
        self.frame_images.pack(pady=5)

        self.label_t1_display = tk.Label(self.frame_images, text="T1 image display area", bg="lightgray")
        self.label_t1_display.pack(side="left", padx=5)

        self.label_t2_display = tk.Label(self.frame_images, text="T2 image display area", bg="lightgray")
        self.label_t2_display.pack(side="left", padx=5)

        # Predict button
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict_mask)
        self.btn_predict.pack(pady=5)

        # Display prediction mask 
        self.label_result = tk.Label(root, text="Predicted Mask will show here", bg="gray")
        self.label_result.pack(pady=5)

        # Pre-process
        self.inference_transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(
                mean=[123.675/255.0, 116.28/255.0, 103.53/255.0],
                std=[58.395/255.0, 57.12/255.0, 57.375/255.0]
            )
        ])

    def load_image_t1(self):
        """Select and present T1"""
        file_path = filedialog.askopenfilename(
            title="Select T1 Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.image_path_t1 = file_path
            # Show T1
            img_pil = Image.open(file_path).convert("RGB")
            img_pil = img_pil.resize(self.display_size, Image.ANTIALIAS)
            self.img_tk_t1 = ImageTk.PhotoImage(img_pil)
            self.label_t1_display.configure(image=self.img_tk_t1, text="")
            self.label_t1_display.image = self.img_tk_t1 

    def load_image_t2(self):
        """Select and present T2"""
        file_path = filedialog.askopenfilename(
            title="Select T2 Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            self.image_path_t2 = file_path
            # Show T2
            img_pil = Image.open(file_path).convert("RGB")
            img_pil = img_pil.resize(self.display_size, Image.ANTIALIAS)
            self.img_tk_t2 = ImageTk.PhotoImage(img_pil)
            self.label_t2_display.configure(image=self.img_tk_t2, text="")
            self.label_t2_display.image = self.img_tk_t2

    def predict_mask(self):
        """Displaying prediction masks"""
        if not self.image_path_t1 or not self.image_path_t2:
            print("Please select both T1 and T2 images before predicting.")
            return

        # Read T1 and T2
        image_t1 = Image.open(self.image_path_t1).convert("RGB")
        image_t2 = Image.open(self.image_path_t2).convert("RGB")

        # Rre-processing
        input_t1 = self.inference_transform(image_t1).unsqueeze(0)
        input_t2 = self.inference_transform(image_t2).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(input_t1, input_t2)  # [1, 1, 512, 512]
            prob = torch.sigmoid(logits)
            pred_mask = (prob > 0.5).float()

        # Convert to PIL Image
        mask_np = pred_mask.squeeze().cpu().numpy() * 255
        mask_img = Image.fromarray(mask_np.astype("uint8"), mode="L")

        # Displayed on screen
        mask_img_tk = ImageTk.PhotoImage(mask_img)
        self.label_result.configure(image=mask_img_tk, text="")
        self.label_result.image = mask_img_tk

def main():
    root = tk.Tk()

    # Calling the trained model
    model_checkpoint = "results/checkpoints/best_model.pth"
    print("Loading ChangeDetectionModel...")

    model = ChangeDetectionModel(
        sam_type="vit_b",
        checkpoint=None,
        freeze_encoder=False
    )

    state_dict = torch.load(model_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    app = SimpleChangeDetectionUI(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
