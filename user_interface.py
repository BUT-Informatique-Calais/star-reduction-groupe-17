import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter
import cv2 as cv

class StarReductionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Star Reduction Tool")
        self.root.geometry("1400x800")
        
        # Variables
        self.original_data = None
        self.luminance = None
        self.current_preview = None
        
        # Default parameters
        self.fwhm_var = tk.DoubleVar(value=3.0)
        self.threshold_var = tk.DoubleVar(value=5.0)
        self.radius_min_var = tk.DoubleVar(value=3.0)
        self.radius_max_var = tk.DoubleVar(value=10.0)
        self.attenuation_var = tk.DoubleVar(value=0.6)
        self.blur_sigma_var = tk.DoubleVar(value=1.5)
        
        # Cache for optimization
        self.detected_sources = None
        self.base_mask = None
        self.update_timer = None
        self.stats_cache = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top frame - File controls
        top_frame = tk.Frame(self.root, bg='#2b2b2b', height=60)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        tk.Button(
            top_frame, 
            text="Load FITS File", 
            command=self.load_fits,
            bg='#4a90e2',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            top_frame, 
            text="Save Result", 
            command=self.save_result,
            bg='#50c878',
            fg='white',
            font=('Arial', 11, 'bold'),
            padx=20,
            pady=8
        ).pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(
            top_frame, 
            text="No file loaded", 
            bg='#2b2b2b',
            fg='#ffffff',
            font=('Arial', 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='#3a3a3a', width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Title
        tk.Label(
            left_panel, 
            text="Reduction Parameters", 
            bg='#3a3a3a',
            fg='#ffffff',
            font=('Arial', 14, 'bold')
        ).pack(pady=15)
        
        # Star Detection Section
        self.create_section(left_panel, "Star Detection")
        
        self.create_slider(
            left_panel,
            "FWHM (star size):",
            self.fwhm_var,
            1.0, 10.0, 0.5,
            self.on_detection_param_change
        )
        
        self.create_slider(
            left_panel,
            "Threshold (σ):",
            self.threshold_var,
            1.0, 15.0, 0.5,
            self.on_detection_param_change
        )
        
        # Mask Section
        self.create_section(left_panel, "Mask Settings")
        
        self.create_slider(
            left_panel,
            "Min Radius:",
            self.radius_min_var,
            1.0, 10.0, 0.5,
            self.schedule_update
        )
        
        self.create_slider(
            left_panel,
            "Max Radius:",
            self.radius_max_var,
            5.0, 30.0, 1.0,
            self.schedule_update
        )
        
        self.create_slider(
            left_panel,
            "Blur Sigma:",
            self.blur_sigma_var,
            0.5, 5.0, 0.1,
            self.schedule_update
        )
        
        # Reduction Section
        self.create_section(left_panel, "Reduction Strength")
        
        self.create_slider(
            left_panel,
            "Attenuation:",
            self.attenuation_var,
            0.0, 1.0, 0.05,
            self.schedule_update
        )
        
        # Process button
        tk.Button(
            left_panel,
            text="Apply Reduction",
            command=self.update_preview,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        ).pack(pady=20)
        
        # Stats label
        self.stats_label = tk.Label(
            left_panel,
            text="",
            bg='#3a3a3a',
            fg='#aaaaaa',
            font=('Arial', 9),
            justify=tk.LEFT
        )
        self.stats_label.pack(pady=10, padx=10)
        
        # Right panel - Image display
        right_panel = tk.Frame(main_container, bg='#1e1e1e')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Image canvas
        self.canvas = tk.Canvas(right_panel, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def create_section(self, parent, title):
        tk.Label(
            parent,
            text=title,
            bg='#3a3a3a',
            fg='#4a90e2',
            font=('Arial', 11, 'bold')
        ).pack(pady=(15, 5), anchor='w', padx=20)
        
    def create_slider(self, parent, label, variable, from_, to, resolution, command):
        frame = tk.Frame(parent, bg='#3a3a3a')
        frame.pack(pady=5, padx=20, fill=tk.X)
        
        label_widget = tk.Label(
            frame,
            text=label,
            bg='#3a3a3a',
            fg='#cccccc',
            font=('Arial', 9)
        )
        label_widget.pack(anchor='w')
        
        value_label = tk.Label(
            frame,
            text=f"{variable.get():.2f}",
            bg='#3a3a3a',
            fg='#ffffff',
            font=('Arial', 9, 'bold')
        )
        value_label.pack(anchor='e')
        
        slider = tk.Scale(
            frame,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=variable,
            bg='#4a4a4a',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#2a2a2a',
            activebackground='#4a90e2',
            command=lambda val: self.on_slider_change(value_label, variable, command)
        )
        slider.pack(fill=tk.X)
        
    def on_slider_change(self, label, variable, callback):
        label.config(text=f"{variable.get():.2f}")
        if callback:
            callback()
    
    def schedule_update(self):
        """Delayed update to avoid recalculating on every slider movement"""
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(300, self.update_preview_fast)
    
    def on_detection_param_change(self):
        """Force full recalculation when detection params change"""
        self.detected_sources = None
        self.base_mask = None
        self.stats_cache = None
        self.schedule_update()
    
    def load_fits(self):
        filename = filedialog.askopenfilename(
            title="Select FITS file",
            filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            hdul = fits.open(filename)
            self.original_data = hdul[0].data.astype(np.float32)
            hdul.close()
            
            # Handle image format
            if self.original_data.ndim == 3:
                if self.original_data.shape[0] == 3:
                    self.original_data = np.transpose(self.original_data, (1, 2, 0))
                self.luminance = np.mean(self.original_data, axis=2)
            else:
                self.luminance = self.original_data.copy()
            
            self.status_label.config(
                text=f"Loaded: {filename.split('/')[-1]} | Shape: {self.original_data.shape}",
                fg='#50c878'
            )
            
            self.display_image(self.normalize_image(self.original_data))
            self.update_preview_fast()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg='#e74c3c')
    
    def normalize_image(self, img):
        if img.ndim == 3:
            lum = np.mean(img, axis=2)
            p1, p99 = np.percentile(lum[lum > 0], (1, 99))
            img_norm = np.zeros_like(img)
            for i in range(3):
                channel = img[:, :, i]
                img_norm[:, :, i] = np.clip((channel - p1) / (p99 - p1), 0, 1)
        else:
            p1, p99 = np.percentile(img[img > 0], (1, 99))
            img_norm = np.clip((img - p1) / (p99 - p1), 0, 1)
        
        return img_norm
    
    def update_preview_fast(self):
        """Optimized preview update with caching"""
        if self.original_data is None:
            return
        
        try:
            # Detect stars only if not cached or params changed
            if self.detected_sources is None:
                mean, median, std = sigma_clipped_stats(self.luminance, sigma=3.0)
                self.stats_cache = (mean, median, std)
                
                finder = DAOStarFinder(
                    fwhm=self.fwhm_var.get(),
                    threshold=self.threshold_var.get() * std
                )
                self.detected_sources = finder(self.luminance - median)
                
                if self.detected_sources is None:
                    self.status_label.config(text="No stars detected", fg='#e74c3c')
                    return
            
            sources = self.detected_sources
            mean, median, std = self.stats_cache
            
            # Create mask (lighter computation)
            mask = np.zeros(self.luminance.shape, dtype=np.float32)
            y_grid, x_grid = np.ogrid[:self.luminance.shape[0], :self.luminance.shape[1]]
            
            for star in sources:
                xc, yc = star["xcentroid"], star["ycentroid"]
                star_flux = star["flux"]
                
                radius = np.clip(
                    self.radius_min_var.get() + star_flux / 10000.0,
                    self.radius_min_var.get(),
                    self.radius_max_var.get()
                )
                
                distance = np.sqrt((x_grid - xc)**2 + (y_grid - yc)**2)
                star_mask = np.maximum(0, 1.0 - distance / radius)
                mask = np.maximum(mask, star_mask)
            
            # Blur mask
            mask = gaussian_filter(mask, sigma=self.blur_sigma_var.get())
            
            # Apply reduction
            attenuation = self.attenuation_var.get()
            
            if self.original_data.ndim == 3:
                data_reduced = self.original_data.copy()
                for i in range(3):
                    channel = self.original_data[:, :, i]
                    bg_channel = gaussian_filter(channel, sigma=20)
                    data_reduced[:, :, i] = (
                        channel * (1.0 - mask * attenuation) + 
                        bg_channel * mask * attenuation * 0.3
                    )
            else:
                bg = gaussian_filter(self.original_data, sigma=20)
                data_reduced = (
                    self.original_data * (1.0 - mask * attenuation) + 
                    bg * mask * attenuation * 0.3
                )
            
            self.current_preview = data_reduced
            self.display_image(self.normalize_image(data_reduced))
            
            # Update stats
            self.stats_label.config(
                text=f"Stars detected: {len(sources)}\n"
                     f"Mean: {mean:.2f}\n"
                     f"Median: {median:.2f}\n"
                     f"Std Dev: {std:.2f}"
            )
            
        except Exception as e:
            self.status_label.config(text=f"Processing error: {str(e)}", fg='#e74c3c')
    
    def update_preview(self):
        """Legacy method - redirects to optimized version"""
        self.update_preview_fast()
    
    def display_image(self, img):
        # Convert to displayable format
        if img.ndim == 3:
            display_img = (img * 255).astype(np.uint8)
        else:
            display_img = (img * 255).astype(np.uint8)
            display_img = cv.cvtColor(display_img, cv.COLOR_GRAY2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = display_img.shape[:2]
            scale = min(canvas_width / w, canvas_height / h) * 0.95
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            display_img = cv.resize(display_img, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.photo,
            anchor=tk.CENTER
        )
    
    def save_result(self):
        if self.current_preview is None:
            self.status_label.config(text="No result to save", fg='#e74c3c')
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if filename:
            normalized = self.normalize_image(self.current_preview)
            if normalized.ndim == 3:
                img_save = (normalized * 255).astype(np.uint8)
            else:
                img_save = (normalized * 255).astype(np.uint8)
            
            cv.imwrite(filename, cv.cvtColor(img_save, cv.COLOR_RGB2BGR))
            self.status_label.config(text=f"Saved: {filename.split('/')[-1]}", fg='#50c878')

if __name__ == "__main__":
    root = tk.Tk()
    app = StarReductionGUI(root)
    root.mainloop()