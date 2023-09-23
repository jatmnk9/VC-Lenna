import matplotlib.pyplot as plt
import cv2
import numpy as np

# Carga la imagen
img = cv2.imread('Lenna.png')

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Guarda la imagen en escala de grises en el disco
cv2.imwrite('gris.jpg', gray)

# Asigna un umbral para determinar los píxeles blancos y negros
threshold = 128
_, binary_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

# Guarda la imagen binaria resultante
cv2.imwrite('binario.jpg', binary_img)

# Calcula el histograma de la imagen binaria
hist = cv2.calcHist([binary_img], [0], None, [256], [0, 256])

# Muestra el histograma de la imagen binaria
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

# Aumentar el contraste
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(binary_img)

# Mostrar imágenes original y con contraste aumentado
cv2.imshow("Original Binary", binary_img)
cv2.imshow("Contrast Increased", cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Mostrar histograma
plt.hist(cl1.ravel(),256,[0,256])
plt.show()
