import sys
from PyQt5.QtWidgets import QApplication, QOpenGLWidget, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class OpenGLWidget(QOpenGLWidget):
    def __init__(self, image_path, parent=None):
        super(OpenGLWidget, self).__init__(parent)
        self.image_path = image_path
        self.texture_id = None
        self.rotate_x = 0
        self.rotate_y = 0

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Load and bind the texture
        image = QImage(self.image_path)
        image = image.convertToFormat(QImage.Format_RGBA8888)
        image = image.mirrored()
        img_data = image.bits().asstring(image.byteCount())
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(self.rotate_x, 1, 0, 0)
        glRotatef(self.rotate_y, 0, 1, 0)

        # Draw the texture on a cube
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(-1, -1, 1)
        glTexCoord2f(1, 0)
        glVertex3f(1, -1, 1)
        glTexCoord2f(1, 1)
        glVertex3f(1, 1, 1)
        glTexCoord2f(0, 1)
        glVertex3f(-1, 1, 1)
        glEnd()

        # self.swapBuffers()

    def mousePressEvent(self, event):
        self.last_x = event.x()
        self.last_y = event.y()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.last_x
        dy = event.y() - self.last_y

        self.rotate_x += dy * 0.5
        self.rotate_y += dx * 0.5

        self.last_x = event.x()
        self.last_y = event.y()

        self.update()

class MainWindow(QWidget):
    def __init__(self, image_path, parent=None):
        super(MainWindow, self).__init__(parent)
        self.image_path = image_path
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        glWidget = OpenGLWidget(self.image_path, self)
        layout.addWidget(glWidget)
        self.setWindowTitle('360-Degree Image Viewer')
        self.setGeometry(100, 100, 800, 600)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    image_path = './output/015_processedOutput.png'  # Change this to the path of your 360-degree image
    window = MainWindow(image_path)
    window.show()
    sys.exit(app.exec_())
