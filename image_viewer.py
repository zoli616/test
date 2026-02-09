import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QPushButton, QFileDialog, QHBoxLayout,
                             QScrollArea, QListWidget, QListWidgetItem, 
                             QLineEdit, QFormLayout, QGroupBox, QColorDialog,
                             QInputDialog, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, pyqtSignal
import cv2
import numpy as np


class Rectangle:
    """Class to represent a drawable rectangle"""
    def __init__(self, x, y, width, height, name="Rectangle", color=None):
        self.rect = QRectF(x, y, width, height)
        self.name = name
        if color is None:
            # Generate random color
            import random
            self.color = QColor(random.randint(100, 255), 
                              random.randint(100, 255), 
                              random.randint(100, 255), 180)
            self.border_color = QColor(self.color.red(), self.color.green(), 
                                      self.color.blue(), 255)
        else:
            self.color = QColor(color[0], color[1], color[2], 180)
            self.border_color = QColor(color[0], color[1], color[2], 255)
        self.selected = False
        self.corner_size = 8  # Size of corner handles
        
    def contains_point(self, point):
        """Check if point is inside rectangle"""
        return self.rect.contains(point)
    
    def get_corner_rect(self, corner, zoom_factor=1.0):
        """Get the rectangle for a corner handle"""
        size = self.corner_size / zoom_factor
        
        if corner == 'top_left':
            return QRectF(self.rect.left() - size/2, self.rect.top() - size/2, size, size)
        elif corner == 'top_right':
            return QRectF(self.rect.right() - size/2, self.rect.top() - size/2, size, size)
        elif corner == 'bottom_left':
            return QRectF(self.rect.left() - size/2, self.rect.bottom() - size/2, size, size)
        elif corner == 'bottom_right':
            return QRectF(self.rect.right() - size/2, self.rect.bottom() - size/2, size, size)
        return None
    
    def get_corner_at_point(self, point, zoom_factor=1.0):
        """Check if point is on any corner, return corner name or None"""
        corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        for corner in corners:
            corner_rect = self.get_corner_rect(corner, zoom_factor)
            if corner_rect.contains(point):
                return corner
        return None
    
    def move(self, delta_x, delta_y):
        """Move the rectangle by delta"""
        self.rect.translate(delta_x, delta_y)
    
    def resize_corner(self, corner, new_x, new_y):
        """Resize rectangle by moving a specific corner"""
        left = self.rect.left()
        top = self.rect.top()
        right = self.rect.right()
        bottom = self.rect.bottom()
        
        if corner == 'top_left':
            left = new_x
            top = new_y
        elif corner == 'top_right':
            right = new_x
            top = new_y
        elif corner == 'bottom_left':
            left = new_x
            bottom = new_y
        elif corner == 'bottom_right':
            right = new_x
            bottom = new_y
        
        # Ensure minimum size
        if right - left < 10:
            if corner in ['top_right', 'bottom_right']:
                right = left + 10
            else:
                left = right - 10
        
        if bottom - top < 10:
            if corner in ['bottom_left', 'bottom_right']:
                bottom = top + 10
            else:
                top = bottom - 10
        
        self.rect = QRectF(left, top, right - left, bottom - top)


class ImageViewer(QLabel):
    # Signal emitted when rectangle list changes or selection changes
    rectangles_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.original_pixmap = None
        self.display_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setMinimumSize(400, 400)
        
        # Enable mouse tracking for cursor position
        self.setMouseTracking(True)
        
        # For panning
        self.last_mouse_pos = None
        self.is_panning = False
        
        # Reference to parent scroll area (will be set later)
        self.scroll_area = None
        
        # Rectangle drawing
        self.rectangles = []
        self.current_rect = None
        self.drawing_rect = False
        self.rect_start_pos = None
        self.rect_counter = 1  # Counter for default names
        
        # Rectangle interaction
        self.selected_rect = None
        self.dragging_rect = False
        self.resizing_corner = None
        self.drag_start_pos = None
        
        # Drawing mode
        self.draw_mode = False
        
    def load_image(self, filepath):
        """Load an image using OpenCV and convert to QPixmap"""
        # Read image with OpenCV
        img = cv2.imread(filepath)
        if img is None:
            return False
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Store original pixmap
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.zoom_factor = 1.0
        self.update_display()
        return True
    
    def update_display(self):
        """Update the displayed image based on current zoom and offset"""
        if self.original_pixmap is None:
            return
            
        # Calculate new size
        new_width = int(self.original_pixmap.width() * self.zoom_factor)
        new_height = int(self.original_pixmap.height() * self.zoom_factor)
        
        # Scale the pixmap
        self.display_pixmap = self.original_pixmap.scaled(
            new_width, new_height, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.setPixmap(self.display_pixmap)
        self.adjustSize()
        self.update()  # Trigger repaint to draw rectangles
    
    def paintEvent(self, event):
        """Override paint event to draw rectangles on top of image"""
        super().paintEvent(event)
        
        if self.display_pixmap is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw all rectangles scaled to current zoom
        for rect_obj in self.rectangles:
            # Scale rectangle coordinates
            scaled_rect = QRectF(
                rect_obj.rect.x() * self.zoom_factor,
                rect_obj.rect.y() * self.zoom_factor,
                rect_obj.rect.width() * self.zoom_factor,
                rect_obj.rect.height() * self.zoom_factor
            )
            
            # Draw filled rectangle with transparency
            painter.setPen(QPen(rect_obj.border_color, 2))
            painter.setBrush(rect_obj.color)
            painter.drawRect(scaled_rect)
            
            # Draw corner handles if selected
            if rect_obj.selected:
                painter.setBrush(QColor(255, 255, 0, 255))  # Yellow handles
                corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
                for corner in corners:
                    # Get corner in original coordinates
                    corner_rect = rect_obj.get_corner_rect(corner, 1.0)
                    # Scale to display coordinates
                    scaled_corner = QRectF(
                        corner_rect.x() * self.zoom_factor,
                        corner_rect.y() * self.zoom_factor,
                        corner_rect.width() * self.zoom_factor,
                        corner_rect.height() * self.zoom_factor
                    )
                    painter.drawRect(scaled_corner)
        
        # Draw current rectangle being drawn
        if self.drawing_rect and self.current_rect:
            scaled_rect = QRectF(
                self.current_rect.rect.x() * self.zoom_factor,
                self.current_rect.rect.y() * self.zoom_factor,
                self.current_rect.rect.width() * self.zoom_factor,
                self.current_rect.rect.height() * self.zoom_factor
            )
            painter.setPen(QPen(QColor(0, 255, 0, 255), 2))
            painter.setBrush(QColor(0, 255, 0, 100))
            painter.drawRect(scaled_rect)
    
    def zoom_at_cursor(self, cursor_pos, zoom_delta):
        """Zoom in/out while keeping cursor position fixed in the image"""
        if self.original_pixmap is None or self.scroll_area is None:
            return
        
        # Get the old zoom factor
        old_zoom = self.zoom_factor
        
        # Update zoom factor
        if zoom_delta > 0:
            self.zoom_factor *= 1.2  # Zoom in
        else:
            self.zoom_factor /= 1.2  # Zoom out
        
        # Limit zoom factor
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
        
        # If zoom didn't actually change, return
        if old_zoom == self.zoom_factor:
            return
        
        # Get scroll bar positions before zoom
        h_scroll = self.scroll_area.horizontalScrollBar()
        v_scroll = self.scroll_area.verticalScrollBar()
        
        old_h_value = h_scroll.value()
        old_v_value = v_scroll.value()
        
        # Get cursor position in the viewport (scroll area)
        viewport_pos = self.scroll_area.viewport().mapFromGlobal(self.mapToGlobal(cursor_pos))
        
        # Calculate the point in the scaled image that cursor is pointing at (before zoom)
        # The absolute position in the scaled image is: scroll_value + viewport_position
        old_image_x = old_h_value + viewport_pos.x()
        old_image_y = old_v_value + viewport_pos.y()
        
        # This point represents a position in the scaled image space
        # Convert to original image coordinates (0-1 normalized)
        old_width = self.original_pixmap.width() * old_zoom
        old_height = self.original_pixmap.height() * old_zoom
        
        # Normalized position in the original image (0.0 to 1.0)
        norm_x = old_image_x / old_width if old_width > 0 else 0.5
        norm_y = old_image_y / old_height if old_height > 0 else 0.5
        
        # Update display with new zoom
        self.update_display()
        
        # Calculate new dimensions
        new_width = self.original_pixmap.width() * self.zoom_factor
        new_height = self.original_pixmap.height() * self.zoom_factor
        
        # Calculate where that normalized point is in the new scaled image
        new_image_x = norm_x * new_width
        new_image_y = norm_y * new_height
        
        # Calculate new scroll values to keep cursor on same image point
        # We want: new_scroll_value + viewport_pos = new_image_position
        new_h_value = new_image_x - viewport_pos.x()
        new_v_value = new_image_y - viewport_pos.y()
        
        # Set new scroll positions
        h_scroll.setValue(int(new_h_value))
        v_scroll.setValue(int(new_v_value))
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_pixmap is None:
            return
            
        # Get the cursor position
        cursor_pos = event.pos()
        
        # Determine zoom direction
        zoom_delta = event.angleDelta().y()
        
        self.zoom_at_cursor(cursor_pos, zoom_delta)
    
    def mousePressEvent(self, event):
        """Handle mouse press for panning, drawing, and selecting rectangles"""
        if self.original_pixmap is None:
            return
        
        # Convert click position to image coordinates (unscaled)
        image_pos = QPoint(int(event.pos().x() / self.zoom_factor),
                          int(event.pos().y() / self.zoom_factor))
        
        if event.button() == Qt.LeftButton:
            if self.draw_mode:
                # Start drawing a new rectangle
                self.drawing_rect = True
                self.rect_start_pos = image_pos
                self.current_rect = Rectangle(image_pos.x(), image_pos.y(), 0, 0)
            else:
                # Check if clicking on a corner of selected rectangle
                if self.selected_rect:
                    corner = self.selected_rect.get_corner_at_point(image_pos, 1.0)
                    if corner:
                        self.resizing_corner = corner
                        self.drag_start_pos = image_pos
                        return
                
                # Check if clicking on any rectangle
                clicked_rect = None
                for rect_obj in reversed(self.rectangles):  # Check from top to bottom
                    if rect_obj.contains_point(image_pos):
                        clicked_rect = rect_obj
                        break
                
                if clicked_rect:
                    # Select and prepare to drag
                    selection_changed = (self.selected_rect != clicked_rect)
                    if self.selected_rect:
                        self.selected_rect.selected = False
                    self.selected_rect = clicked_rect
                    self.selected_rect.selected = True
                    self.dragging_rect = True
                    self.drag_start_pos = image_pos
                    self.update()
                    if selection_changed:
                        self.rectangles_changed.emit()
                else:
                    # Deselect if clicking empty space
                    if self.selected_rect:
                        self.selected_rect.selected = False
                        self.selected_rect = None
                        self.update()
                        self.rectangles_changed.emit()
                    # Start panning
                    self.is_panning = True
                    self.last_mouse_pos = event.pos()
                    self.setCursor(Qt.ClosedHandCursor)
        
        elif event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning, drawing, dragging, and resizing rectangles"""
        if self.original_pixmap is None:
            return
        
        # Convert position to image coordinates
        image_pos = QPoint(int(event.pos().x() / self.zoom_factor),
                          int(event.pos().y() / self.zoom_factor))
        
        # Handle rectangle drawing
        if self.drawing_rect and self.current_rect and self.rect_start_pos:
            x = min(self.rect_start_pos.x(), image_pos.x())
            y = min(self.rect_start_pos.y(), image_pos.y())
            width = abs(image_pos.x() - self.rect_start_pos.x())
            height = abs(image_pos.y() - self.rect_start_pos.y())
            self.current_rect.rect = QRectF(x, y, width, height)
            self.update()
            return
        
        # Handle rectangle resizing
        if self.resizing_corner and self.selected_rect and self.drag_start_pos:
            self.selected_rect.resize_corner(self.resizing_corner, image_pos.x(), image_pos.y())
            self.update()
            return
        
        # Handle rectangle dragging
        if self.dragging_rect and self.selected_rect and self.drag_start_pos:
            delta_x = image_pos.x() - self.drag_start_pos.x()
            delta_y = image_pos.y() - self.drag_start_pos.y()
            self.selected_rect.move(delta_x, delta_y)
            self.drag_start_pos = image_pos
            self.update()
            return
        
        # Handle panning
        if self.is_panning and self.last_mouse_pos is not None:
            # Calculate delta movement
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            
            # Update scroll bars
            if self.scroll_area:
                h_scroll = self.scroll_area.horizontalScrollBar()
                v_scroll = self.scroll_area.verticalScrollBar()
                h_scroll.setValue(h_scroll.value() - delta.x())
                v_scroll.setValue(v_scroll.value() - delta.y())
            return
        
        # Update cursor based on what's under it
        if not self.draw_mode and self.selected_rect:
            corner = self.selected_rect.get_corner_at_point(image_pos, 1.0)
            if corner in ['top_left', 'bottom_right']:
                self.setCursor(Qt.SizeFDiagCursor)
            elif corner in ['top_right', 'bottom_left']:
                self.setCursor(Qt.SizeBDiagCursor)
            elif self.selected_rect.contains_point(image_pos):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif self.draw_mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton:
            # Finish drawing rectangle
            if self.drawing_rect and self.current_rect:
                if self.current_rect.rect.width() > 5 and self.current_rect.rect.height() > 5:
                    # Assign a default name
                    self.current_rect.name = f"Rectangle {self.rect_counter}"
                    self.rect_counter += 1
                    self.rectangles.append(self.current_rect)
                    self.rectangles_changed.emit()
                self.drawing_rect = False
                self.current_rect = None
                self.rect_start_pos = None
                self.update()
            
            # Finish dragging or resizing
            if self.dragging_rect:
                self.dragging_rect = False
                self.drag_start_pos = None
                self.rectangles_changed.emit()
            
            if self.resizing_corner:
                self.resizing_corner = None
                self.drag_start_pos = None
                self.rectangles_changed.emit()
            
            # Stop panning
            if self.is_panning:
                self.is_panning = False
                self.setCursor(Qt.ArrowCursor)
        
        elif event.button() == Qt.MiddleButton:
            if self.is_panning:
                self.is_panning = False
                self.setCursor(Qt.ArrowCursor)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Zoom and Annotations")
        self.setGeometry(100, 100, 1200, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create button layout
        button_layout = QHBoxLayout()
        
        # Open button
        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self.open_image)
        button_layout.addWidget(open_btn)
        
        # Zoom in button
        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_in_btn.clicked.connect(self.zoom_in)
        button_layout.addWidget(zoom_in_btn)
        
        # Zoom out button
        zoom_out_btn = QPushButton("Zoom Out (-)")
        zoom_out_btn.clicked.connect(self.zoom_out)
        button_layout.addWidget(zoom_out_btn)
        
        # Reset button
        reset_btn = QPushButton("Reset Zoom")
        reset_btn.clicked.connect(self.reset_zoom)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        # Zoom label
        self.zoom_label = QLabel("Zoom: 100%")
        button_layout.addWidget(self.zoom_label)
        
        # Add spacing
        button_layout.addSpacing(20)
        
        # Draw mode toggle button
        self.draw_mode_btn = QPushButton("Draw Mode: OFF")
        self.draw_mode_btn.setCheckable(True)
        self.draw_mode_btn.clicked.connect(self.toggle_draw_mode)
        self.draw_mode_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        button_layout.addWidget(self.draw_mode_btn)
        
        # Delete rectangle button
        delete_rect_btn = QPushButton("Delete Selected")
        delete_rect_btn.clicked.connect(self.delete_selected_rectangle)
        button_layout.addWidget(delete_rect_btn)
        
        # Clear all rectangles button
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_rectangles)
        button_layout.addWidget(clear_all_btn)
        
        main_layout.addLayout(button_layout)
        
        # Create splitter for image viewer and rectangle list
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Image viewer in scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignCenter)
        
        # Create image viewer
        self.image_viewer = ImageViewer()
        self.image_viewer.scroll_area = scroll_area
        scroll_area.setWidget(self.image_viewer)
        
        splitter.addWidget(scroll_area)
        
        # Right side: Rectangle list panel
        self.rectangle_panel = self.create_rectangle_panel()
        splitter.addWidget(self.rectangle_panel)
        
        # Set initial sizes (image viewer gets 70%, panel gets 30%)
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter)
        
        # Connect signals
        self.image_viewer.rectangles_changed.connect(self.update_rectangle_list)
    
    def create_rectangle_panel(self):
        """Create the rectangle list panel with properties"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Rectangles")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Rectangle list
        self.rect_list = QListWidget()
        self.rect_list.itemClicked.connect(self.on_rectangle_list_clicked)
        layout.addWidget(self.rect_list)
        
        # Properties group
        properties_group = QGroupBox("Properties")
        properties_layout = QFormLayout()
        
        # Name field
        self.name_field = QLineEdit()
        self.name_field.textChanged.connect(self.on_name_changed)
        properties_layout.addRow("Name:", self.name_field)
        
        # Color button
        self.color_btn = QPushButton("Change Color")
        self.color_btn.clicked.connect(self.change_rectangle_color)
        properties_layout.addRow("Color:", self.color_btn)
        
        # Coordinates
        self.x_field = QLineEdit()
        self.x_field.textChanged.connect(self.on_coordinate_changed)
        properties_layout.addRow("X:", self.x_field)
        
        self.y_field = QLineEdit()
        self.y_field.textChanged.connect(self.on_coordinate_changed)
        properties_layout.addRow("Y:", self.y_field)
        
        self.width_field = QLineEdit()
        self.width_field.textChanged.connect(self.on_coordinate_changed)
        properties_layout.addRow("Width:", self.width_field)
        
        self.height_field = QLineEdit()
        self.height_field.textChanged.connect(self.on_coordinate_changed)
        properties_layout.addRow("Height:", self.height_field)
        
        properties_group.setLayout(properties_layout)
        layout.addWidget(properties_group)
        
        # Initially disable properties
        self.set_properties_enabled(False)
        
        return panel
    
    def set_properties_enabled(self, enabled):
        """Enable or disable property fields"""
        self.name_field.setEnabled(enabled)
        self.color_btn.setEnabled(enabled)
        self.x_field.setEnabled(enabled)
        self.y_field.setEnabled(enabled)
        self.width_field.setEnabled(enabled)
        self.height_field.setEnabled(enabled)
    
    def update_rectangle_list(self):
        """Update the rectangle list widget"""
        # Block signals to prevent triggering selection changes
        self.rect_list.blockSignals(True)
        
        # Remember current selection
        current_rect = self.image_viewer.selected_rect
        
        # Clear and repopulate list
        self.rect_list.clear()
        
        for i, rect_obj in enumerate(self.image_viewer.rectangles):
            # Create list item with rectangle info
            x = int(rect_obj.rect.x())
            y = int(rect_obj.rect.y())
            w = int(rect_obj.rect.width())
            h = int(rect_obj.rect.height())
            
            item_text = f"{rect_obj.name} ({x}, {y}, {w}, {h})"
            item = QListWidgetItem(item_text)
            
            # Set background color to match rectangle
            color = rect_obj.border_color
            item.setBackground(QColor(color.red(), color.green(), color.blue(), 50))
            
            # Store reference to rectangle object
            item.setData(Qt.UserRole, rect_obj)
            
            self.rect_list.addItem(item)
            
            # Select this item if it's the selected rectangle
            if rect_obj == current_rect:
                item.setSelected(True)
        
        self.rect_list.blockSignals(False)
        
        # Update properties panel
        self.update_properties_panel()
    
    def update_properties_panel(self):
        """Update the properties panel based on selected rectangle"""
        if self.image_viewer.selected_rect:
            self.set_properties_enabled(True)
            rect = self.image_viewer.selected_rect
            
            # Block signals to prevent infinite loops
            self.name_field.blockSignals(True)
            self.x_field.blockSignals(True)
            self.y_field.blockSignals(True)
            self.width_field.blockSignals(True)
            self.height_field.blockSignals(True)
            
            self.name_field.setText(rect.name)
            self.x_field.setText(str(int(rect.rect.x())))
            self.y_field.setText(str(int(rect.rect.y())))
            self.width_field.setText(str(int(rect.rect.width())))
            self.height_field.setText(str(int(rect.rect.height())))
            
            # Update color button
            color = rect.border_color
            self.color_btn.setStyleSheet(f"""
                background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                color: white;
            """)
            
            self.name_field.blockSignals(False)
            self.x_field.blockSignals(False)
            self.y_field.blockSignals(False)
            self.width_field.blockSignals(False)
            self.height_field.blockSignals(False)
        else:
            self.set_properties_enabled(False)
            self.name_field.clear()
            self.x_field.clear()
            self.y_field.clear()
            self.width_field.clear()
            self.height_field.clear()
            self.color_btn.setStyleSheet("")
    
    def on_rectangle_list_clicked(self, item):
        """Handle clicking on rectangle in the list"""
        rect_obj = item.data(Qt.UserRole)
        if rect_obj:
            # Deselect current
            if self.image_viewer.selected_rect:
                self.image_viewer.selected_rect.selected = False
            
            # Select new rectangle
            self.image_viewer.selected_rect = rect_obj
            rect_obj.selected = True
            self.image_viewer.update()
            self.update_properties_panel()
    
    def on_name_changed(self):
        """Handle name field change"""
        if self.image_viewer.selected_rect:
            self.image_viewer.selected_rect.name = self.name_field.text()
            self.update_rectangle_list()
    
    def on_coordinate_changed(self):
        """Handle coordinate field changes"""
        if not self.image_viewer.selected_rect:
            return
        
        try:
            x = float(self.x_field.text())
            y = float(self.y_field.text())
            w = float(self.width_field.text())
            h = float(self.height_field.text())
            
            # Update rectangle
            self.image_viewer.selected_rect.rect = QRectF(x, y, w, h)
            self.image_viewer.update()
            self.update_rectangle_list()
        except ValueError:
            # Invalid number, ignore
            pass
    
    def change_rectangle_color(self):
        """Open color picker to change rectangle color"""
        if not self.image_viewer.selected_rect:
            return
        
        current_color = self.image_viewer.selected_rect.border_color
        color = QColorDialog.getColor(current_color, self, "Select Rectangle Color")
        
        if color.isValid():
            self.image_viewer.selected_rect.color = QColor(color.red(), color.green(), 
                                                          color.blue(), 180)
            self.image_viewer.selected_rect.border_color = color
            self.image_viewer.update()
            self.update_rectangle_list()
            self.update_properties_panel()
    
    def open_image(self):
        """Open an image file dialog"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)"
        )
        
        if filepath:
            if self.image_viewer.load_image(filepath):
                self.update_zoom_label()
                self.statusBar().showMessage(f"Loaded: {filepath}")
            else:
                self.statusBar().showMessage("Failed to load image")
    
    def zoom_in(self):
        """Zoom in using center of view"""
        if self.image_viewer.original_pixmap is not None:
            center = self.image_viewer.rect().center()
            self.image_viewer.zoom_at_cursor(center, 120)
            self.update_zoom_label()
    
    def zoom_out(self):
        """Zoom out using center of view"""
        if self.image_viewer.original_pixmap is not None:
            center = self.image_viewer.rect().center()
            self.image_viewer.zoom_at_cursor(center, -120)
            self.update_zoom_label()
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        if self.image_viewer.original_pixmap is not None:
            self.image_viewer.zoom_factor = 1.0
            self.image_viewer.update_display()
            self.update_zoom_label()
    
    def update_zoom_label(self):
        """Update the zoom percentage label"""
        zoom_percent = int(self.image_viewer.zoom_factor * 100)
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")
    
    def toggle_draw_mode(self):
        """Toggle rectangle drawing mode"""
        self.image_viewer.draw_mode = self.draw_mode_btn.isChecked()
        if self.image_viewer.draw_mode:
            self.draw_mode_btn.setText("Draw Mode: ON")
            self.image_viewer.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage("Draw mode ON - Click and drag to draw rectangles")
        else:
            self.draw_mode_btn.setText("Draw Mode: OFF")
            self.image_viewer.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("Draw mode OFF")
    
    def delete_selected_rectangle(self):
        """Delete the currently selected rectangle"""
        if self.image_viewer.selected_rect:
            self.image_viewer.rectangles.remove(self.image_viewer.selected_rect)
            self.image_viewer.selected_rect = None
            self.image_viewer.update()
            self.update_rectangle_list()
            self.statusBar().showMessage("Rectangle deleted")
        else:
            self.statusBar().showMessage("No rectangle selected")
    
    def clear_all_rectangles(self):
        """Clear all rectangles"""
        self.image_viewer.rectangles.clear()
        self.image_viewer.selected_rect = None
        self.image_viewer.rect_counter = 1
        self.image_viewer.update()
        self.update_rectangle_list()
        self.statusBar().showMessage("All rectangles cleared")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
