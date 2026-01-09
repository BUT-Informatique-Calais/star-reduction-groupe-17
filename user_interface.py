from statistics import mode
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
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
        self.show_original = False
        
        # Default parameters
        self.fwhm_var = tk.DoubleVar(value=3.0)
        self.threshold_var = tk.DoubleVar(value=5.0)
        self.erosion_iter_var = tk.IntVar(value=6)
        self.ratio_min_var = tk.DoubleVar(value=0.25)
        self.stretch_var = tk.DoubleVar(value=1.0)
        self.black_point_var = tk.DoubleVar(value=1.0)
        
        # Multi-size reduction parameters
        self.multi_size_enabled = tk.BooleanVar(value=False)
        self.small_erosion_var = tk.IntVar(value=3)
        self.medium_erosion_var = tk.IntVar(value=6)
        self.large_erosion_var = tk.IntVar(value=9)
        self.small_threshold_var = tk.DoubleVar(value=0.3)
        self.large_threshold_var = tk.DoubleVar(value=0.7)
        
        # Cache for optimization
        self.detected_sources = None
        self.update_timer = None
        self.stats_cache = None
        
        # Comparison mode
        self.comparison_mode = tk.StringVar(value="result")
        self.blink_running = False
        self.blink_state = False
        
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
        
        # Comparison mode buttons
        comparison_frame = tk.Frame(top_frame, bg='#2b2b2b')
        comparison_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            comparison_frame,
            text="View:",
            bg='#2b2b2b',
            fg='#ffffff',
            font=('Arial', 9, 'bold')
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            comparison_frame,
            text="Original",
            command=lambda: self.set_view_mode("original"),
            bg='#555555',
            fg='white',
            font=('Arial', 9),
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            comparison_frame,
            text="Result",
            command=lambda: self.set_view_mode("result"),
            bg='#555555',
            fg='white',
            font=('Arial', 9),
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            comparison_frame,
            text="Side by Side",
            command=lambda: self.set_view_mode("side_by_side"),
            bg='#555555',
            fg='white',
            font=('Arial', 9),
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            comparison_frame,
            text="Blink",
            command=self.toggle_blink,
            bg='#555555',
            fg='white',
            font=('Arial', 9),
            padx=15,
            pady=5
        ).pack(side=tk.LEFT, padx=2)
        
        self.status_label = tk.Label(
            top_frame, 
            text="No file loaded", 
            bg='#2b2b2b',
            fg='#ffffff',
            font=('Arial', 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.view_label = tk.Label(
            top_frame,
            text="View: RESULT",
            bg='#2b2b2b',
            fg='#4a90e2',
            font=('Arial', 10, 'bold')
        )
        self.view_label.pack(side=tk.RIGHT, padx=10)

        
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Controls (scrollable)
        scrollable_frame = tk.Frame(main_container, bg='#3a3a3a', width=350)
        scrollable_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        scrollable_frame.pack_propagate(False)

        left_canvas = tk.Canvas(
            scrollable_frame,
            bg='#3a3a3a',
            highlightthickness=0
        )
        left_scrollbar = ttk.Scrollbar(
            scrollable_frame,
            orient="vertical",
            command=left_canvas.yview
        )
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollable_inner_frame = tk.Frame(left_canvas, bg='#3a3a3a')

        left_canvas.create_window(
            (0, 0),
            window=scrollable_inner_frame,
            anchor="nw"
        )

        scrollable_inner_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(
                scrollregion=left_canvas.bbox("all")
            )
        )
        # Scroll à la molette
        left_canvas.bind_all(
            "<MouseWheel>",
            lambda e: left_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        )
        
        # Title
        tk.Label(
            scrollable_inner_frame, 
            text="Reduction Parameters", 
            bg='#3a3a3a',
            fg='#ffffff',
            font=('Arial', 14, 'bold')
        ).pack(pady=15)
        
        # Star Detection Section
        self.create_section(scrollable_inner_frame, "Star Detection")
        
        self.create_slider(
            scrollable_inner_frame,
            "FWHM (star size):",
            self.fwhm_var,
            1.0, 10.0, 0.5,
            self.on_detection_param_change
        )
        
        self.create_slider(
            scrollable_inner_frame,
            "Threshold (σ):",
            self.threshold_var,
            1.0, 15.0, 0.5,
            self.on_detection_param_change
        )
        
        # Reduction Section
        self.create_section(scrollable_inner_frame, "Reduction Settings")
        
        self.create_slider(
            scrollable_inner_frame,
            "Erosion Iterations:",
            self.erosion_iter_var,
            1, 12, 1,
            self.schedule_update
        )
        
        self.create_slider(
            scrollable_inner_frame,
            "Ratio Min:",
            self.ratio_min_var,
            0.0, 1.0, 0.05,
            self.schedule_update
        )
        
        # Multi-Size Reduction Section
        self.create_section(scrollable_inner_frame, "Multi-Size Reduction")
        
        # Checkbox to enable/disable multi-size
        multi_size_frame = tk.Frame(scrollable_inner_frame, bg='#3a3a3a')
        multi_size_frame.pack(pady=5, padx=20, fill=tk.X)
        
        tk.Checkbutton(
            multi_size_frame,
            text="Enable Multi-Size Reduction",
            variable=self.multi_size_enabled,
            bg='#3a3a3a',
            fg='#cccccc',
            selectcolor='#4a4a4a',
            activebackground='#3a3a3a',
            activeforeground='#ffffff',
            font=('Arial', 10, 'bold'),
            command=self.schedule_update
        ).pack(anchor='w')
        
        tk.Label(
            scrollable_inner_frame,
            text="Adjust erosion levels per star size:",
            bg='#3a3a3a',
            fg='#999999',
            font=('Arial', 8, 'italic')
        ).pack(pady=(5, 10), padx=20, anchor='w')
        
        self.create_slider(
            scrollable_inner_frame,
            "Small Stars (erosion):",
            self.small_erosion_var,
            1, 8, 1,
            self.schedule_update
        )
        
        self.create_slider(
            scrollable_inner_frame,
            "Medium Stars (erosion):",
            self.medium_erosion_var,
            3, 10, 1,
            self.schedule_update
        )
        
        self.create_slider(
            scrollable_inner_frame,
            "Large Stars (erosion):",
            self.large_erosion_var,
            5, 15, 1,
            self.schedule_update
        )
        
        self.create_slider(
            scrollable_inner_frame,
            "Small/Medium Threshold:",
            self.small_threshold_var,
            0.1, 0.5, 0.05,
            self.schedule_update
        )
        
        self.create_slider(
            scrollable_inner_frame,
            "Medium/Large Threshold:",
            self.large_threshold_var,
            0.5, 0.9, 0.05,
            self.schedule_update
        )
        
        # Display Section
        self.create_section(scrollable_inner_frame, "Display Adjustments")
        
        self.create_slider(
            scrollable_inner_frame,
            "Black Point (%):",
            self.black_point_var,
            0.0, 10.0, 0.1,
            self.schedule_display_update
        )
        
        self.create_slider(
            scrollable_inner_frame,
            "Stretch (Gamma):",
            self.stretch_var,
            0.1, 3.0, 0.1,
            self.schedule_display_update
        )
        
        # Process button
        tk.Button(
            scrollable_inner_frame,
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
            scrollable_inner_frame,
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
        
        if isinstance(variable, tk.IntVar):
            value_label = tk.Label(
                frame,
                text=f"{variable.get()}",
                bg='#3a3a3a',
                fg='#ffffff',
                font=('Arial', 9, 'bold')
            )
        else:
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
        if isinstance(variable, tk.IntVar):
            label.config(text=f"{variable.get()}")
        else:
            label.config(text=f"{variable.get():.2f}")
        if callback:
            callback()
    
    def schedule_update(self):
        """Delayed update to avoid recalculating on every slider movement"""
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(300, self.update_preview_fast)
    
    def schedule_display_update(self):
        """Update display only (no reprocessing)"""
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(100, self.update_display)
    
    def on_detection_param_change(self):
        """Force full recalculation when detection params change"""
        self.detected_sources = None
        self.stats_cache = None
        self.schedule_update()
    
    def set_view_mode(self, mode):
        """Change view mode"""
        if self.blink_running:
            self.blink_running = False

        self.comparison_mode.set(mode)

        if mode == "original":
            self.view_label.config(text="View: ORIGINAL", fg="#e67e22")
        elif mode == "result":
            self.view_label.config(text="View: RESULT", fg="#4a90e2")
        elif mode == "side_by_side":
            self.view_label.config(text="View: BEFORE / AFTER", fg="#9b59b6")
        elif mode == "blink":
            self.view_label.config(text="View: BLINK", fg="#f1c40f")

        self.update_display()

    
    def toggle_blink(self):
        """Toggle blink animation"""
        self.blink_running = not self.blink_running
        if self.blink_running:
            self.comparison_mode.set("blink")
            self.view_label.config(text="View: BLINK", fg="#f1c40f")
            self.blink_animation()

    
    def blink_animation(self):
        """Animate blinking between original and result"""
        if not self.blink_running or self.original_data is None:
            return
        
        self.blink_state = not self.blink_state
        
        if self.blink_state and self.current_preview is not None:
            self.display_image(self.normalize_image(self.current_preview))
        else:
            self.display_image(self.normalize_image(self.original_data))
        
        self.root.after(500, self.blink_animation)
    
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
                R, G, B = self.original_data[:, :, 0], self.original_data[:, :, 1], self.original_data[:, :, 2]
                self.luminance = (R + G + B) / 3.0
            else:
                self.luminance = self.original_data.copy()
            
            self.status_label.config(
                text=f"Loaded: {filename.split('/')[-1]} | Shape: {self.original_data.shape}",
                fg='#50c878'
            )
            
            self.update_display()
            self.update_preview_fast()
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg='#e74c3c')
    
    def normalize_image(self, img):
        """Normalize image with percentile clipping and adjustable stretch"""
        if img.ndim == 3:
            # For color images, normalize based on combined luminance
            lum = np.mean(img, axis=2)
            # Use percentiles to handle extreme values
            p_low = np.percentile(lum[lum > 0], self.black_point_var.get())
            p_high = np.percentile(lum[lum > 0], 99.5)
            
            img_norm = np.zeros_like(img)
            for i in range(3):
                channel = img[:, :, i]
                # Clip and normalize each channel
                normalized = np.clip((channel - p_low) / (p_high - p_low + 1e-6), 0, 1)
                # Apply gamma stretch
                img_norm[:, :, i] = np.power(normalized, 1.0 / self.stretch_var.get())
        else:
            # For monochrome images
            p_low = np.percentile(img[img > 0], self.black_point_var.get())
            p_high = np.percentile(img[img > 0], 99.5)
            normalized = np.clip((img - p_low) / (p_high - p_low + 1e-6), 0, 1)
            # Apply gamma stretch
            img_norm = np.power(normalized, 1.0 / self.stretch_var.get())
        
        return img_norm
    
    def update_preview_fast(self):
        """Optimized preview update with color-preserving reduction"""
        if self.original_data is None:
            return
        
        try:
            # Detect stars only if not cached
            if self.detected_sources is None:
                mean, median, std = sigma_clipped_stats(self.luminance, sigma=3.0)
                self.stats_cache = (mean, median, std)
                
                finder = DAOStarFinder(
                    fwhm=self.fwhm_var.get(),
                    threshold=self.threshold_var.get() * std
                )
                self.detected_sources = finder(self.luminance - median)
                
                if self.detected_sources is None or len(self.detected_sources) == 0:
                    self.status_label.config(text="No stars detected - try adjusting threshold", fg='#ff9800')
                    # Return original image if no stars detected
                    self.current_preview = self.original_data.copy()
                    self.update_display()
                    return
            
            sources = self.detected_sources
            mean, median, std = self.stats_cache
            
            # Normalize luminance for reduction using percentiles
            p_low, p_high = np.percentile(self.luminance[self.luminance > 0], (1, 99.5))
            L_norm = np.clip((self.luminance - p_low) / (p_high - p_low + 1e-6), 0, 1)
            L_uint8 = (L_norm * 255).astype(np.uint8)
            L_reduced = L_uint8.copy()
            
            # Multi-size reduction: calculate flux percentiles for categorization
            if self.multi_size_enabled.get() and len(sources) > 0:
                fluxes = np.array([s["flux"] for s in sources])
                small_threshold = np.percentile(fluxes, self.small_threshold_var.get() * 100)
                large_threshold = np.percentile(fluxes, self.large_threshold_var.get() * 100)
                
                star_counts = {'small': 0, 'medium': 0, 'large': 0}
            
            # Apply erosion on each star
            for star in sources:
                x = int(star["xcentroid"])
                y = int(star["ycentroid"])
                flux = star["flux"]
                
                # Determine erosion iterations based on flux (if multi-size enabled)
                if self.multi_size_enabled.get():
                    if flux < small_threshold:
                        erosion_iters = self.small_erosion_var.get()
                        star_counts['small'] += 1
                    elif flux < large_threshold:
                        erosion_iters = self.medium_erosion_var.get()
                        star_counts['medium'] += 1
                    else:
                        erosion_iters = self.large_erosion_var.get()
                        star_counts['large'] += 1
                else:
                    erosion_iters = self.erosion_iter_var.get()
                
                diameter = int(np.clip(2.0 * np.sqrt(flux), 3, 25))
                if diameter % 2 == 0:
                    diameter += 1
                
                half = diameter // 2
                y1, y2 = max(0, y - half), min(L_reduced.shape[0], y + half + 1)
                x1, x2 = max(0, x - half), min(L_reduced.shape[1], x + half + 1)
                
                patch = L_reduced[y1:y2, x1:x2].copy()
                
                kernel_size = max(3, diameter // 2)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                for _ in range(erosion_iters):
                    patch = cv.erode(patch, kernel, iterations=1)
                    if np.count_nonzero(patch > 15) <= 1:
                        break
                
                L_reduced[y1:y2, x1:x2] = patch
            
            # Calculate ratio for color preservation
            L_reduced_f = L_reduced.astype(np.float32) / 255.0
            L_orig_f = L_uint8.astype(np.float32) / 255.0
            
            ratio = L_reduced_f / (L_orig_f + 1e-6)
            ratio = np.clip(ratio, self.ratio_min_var.get(), 1.0)
            ratio = cv.GaussianBlur(ratio, (5, 5), 0)
            
            # Apply ratio to preserve colors
            if self.original_data.ndim == 3:
                out = np.zeros_like(self.original_data)
                out[:, :, 0] = self.original_data[:, :, 0] * ratio
                out[:, :, 1] = self.original_data[:, :, 1] * ratio
                out[:, :, 2] = self.original_data[:, :, 2] * ratio
            else:
                out = self.luminance * ratio
            
            self.current_preview = out
            self.update_display()
            
            # Update stats
            stats_text = f"Stars detected: {len(sources)}\n"
            stats_text += f"Mean: {mean:.2f}\n"
            stats_text += f"Median: {median:.2f}\n"
            stats_text += f"Std Dev: {std:.2f}"
            
            if self.multi_size_enabled.get():
                stats_text += f"\n\nStar Size Distribution:\n"
                stats_text += f"Small: {star_counts['small']}\n"
                stats_text += f"Medium: {star_counts['medium']}\n"
                stats_text += f"Large: {star_counts['large']}"
            
            self.stats_label.config(text=stats_text)
            
        except Exception as e:
            self.status_label.config(text=f"Processing error: {str(e)}", fg='#e74c3c')
    
    def update_preview(self):
        """Legacy method - redirects to optimized version"""
        self.update_preview_fast()
    
    def update_display(self):
        """Update display based on current mode"""
        if self.original_data is None:
            return
        
        mode = self.comparison_mode.get()
        
        if mode == "original":
            self.display_image(self.normalize_image(self.original_data))
        elif mode == "result":
            if self.current_preview is not None:
                self.display_image(self.normalize_image(self.current_preview))
            else:
                self.display_image(self.normalize_image(self.original_data))
        elif mode == "side_by_side":
            self.display_side_by_side()
    
    def display_side_by_side(self):
        """Display original and processed images side by side"""
        if self.original_data is None:
            return
        
        original_norm = self.normalize_image(self.original_data)
        
        if self.current_preview is not None:
            processed_norm = self.normalize_image(self.current_preview)
        else:
            processed_norm = original_norm
        
        # Convert to displayable format
        if original_norm.ndim == 3:
            orig_display = (original_norm * 255).astype(np.uint8)
            proc_display = (processed_norm * 255).astype(np.uint8)
        else:
            orig_display = (original_norm * 255).astype(np.uint8)
            proc_display = (processed_norm * 255).astype(np.uint8)
            orig_display = cv.cvtColor(orig_display, cv.COLOR_GRAY2RGB)
            proc_display = cv.cvtColor(proc_display, cv.COLOR_GRAY2RGB)
        
        # Resize to fit half canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = orig_display.shape[:2]
            scale = min((canvas_width / 2) / w, canvas_height / h) * 0.9
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            orig_display = cv.resize(orig_display, (new_w, new_h), interpolation=cv.INTER_AREA)
            proc_display = cv.resize(proc_display, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        # Add labels
        orig_labeled = self.add_label_to_image(orig_display, "BEFORE")
        proc_labeled = self.add_label_to_image(proc_display, "AFTER")
        
        # Combine side by side
        combined = np.hstack([orig_labeled, proc_labeled])
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(combined)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.photo,
            anchor=tk.CENTER
        )
    
    def add_label_to_image(self, img, text):
        """Add a label at the top of an image"""
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        
        # Add white rectangle at top
        cv.rectangle(img_copy, (0, 0), (w, 40), (255, 255, 255), -1)
        
        # Add text
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 28
        
        cv.putText(img_copy, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        return img_copy
    
    def display_image(self, img):
        """Display an image on the canvas"""
        if img is None:
            return

        # Convert to uint8 RGB for display
        if img.ndim == 3:
            display_img = np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            display_img = np.clip(img * 255, 0, 255).astype(np.uint8)
            display_img = cv.cvtColor(display_img, cv.COLOR_GRAY2RGB)
        # Canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 2 or canvas_height < 2:
            return
        # Resize to fit canvas
        h, w = display_img.shape[:2]
        scale = min(canvas_width / w, canvas_height / h) * 0.95
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = cv.resize(display_img, (new_w, new_h), interpolation=cv.INTER_AREA)

        # Convert to PhotoImage
        img_pil = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(img_pil)
        # Display
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
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

        if not filename:
            return
        normalized = self.normalize_image(self.current_preview)
        img_save = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        if img_save.ndim == 3:
            img_save = cv.cvtColor(img_save, cv.COLOR_RGB2BGR)
        
        cv.imwrite(filename, img_save)
        self.status_label.config(
            text=f"Saved: {filename.split('/')[-1]}",
            fg='#50c878'
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = StarReductionGUI(root)
    root.mainloop()