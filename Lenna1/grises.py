import matplotlib.pyplot as plt
import cv2

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

# Recorta el primer cuadrante de la imagen
height, width = img.shape[:2]
first_quadrant = img[:height//2, :width//2]

# Muestra la imagen recortada
cv2.imwrite('recorte.jpg', first_quadrant)



