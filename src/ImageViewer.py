import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageViewer(QWidget):
    def __init__(self, folder):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        # Load image paths
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.images = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                       if f.lower().endswith(exts)]
        if not self.images:
            raise ValueError("No images found in folder!")

        self.index = 0

        # UI elements
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")

        self.btn_left.clicked.connect(self.show_prev)
        self.btn_right.clicked.connect(self.show_next)

        # Layout
        hbox = QHBoxLayout()
        hbox.addWidget(self.btn_left)
        hbox.addWidget(self.btn_right)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # Show first image
        self.show_image()

    def show_image(self):
        path = self.images[self.index]
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.label.setPixmap(pixmap.scaled(
                self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        self.setWindowTitle(f"Image Viewer - {os.path.basename(path)}")

    def resizeEvent(self, event):
        self.show_image()

    def show_next(self):
        self.index = (self.index + 1) % len(self.images)
        self.show_image()

    def show_prev(self):
        self.index = (self.index - 1) % len(self.images)
        self.show_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Pick folder via dialog (or hardcode a path)
    folder = QFileDialog.getExistingDirectory(None, "Select Image Folder")
    if folder:
        viewer = ImageViewer(folder)
        viewer.resize(800, 600)
        viewer.show()
        sys.exit(app.exec_())