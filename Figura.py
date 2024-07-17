import cv2
import face_recognition as fr
import numpy as np
from datetime import datetime
import math

class SistemaReconocimientoFacial:
    def __init__(self):
        self.rostros_codificados = []
        self.nombres_rostros = []

    def agregar_rostro(self, imagen, nombre):
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificaciones = fr.face_encodings(imagen_rgb)[0]
        self.rostros_codificados.append(codificaciones)
        self.nombres_rostros.append(nombre)

    def tomar_foto(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        exito, img = cap.read()
        cap.release()
        if not exito:
            raise Exception("No se pudo tomar la foto")
        return img

    def reconocer_rostro(self, img):
        ubicaciones_rostros = fr.face_locations(img)
        codificaciones_rostros = fr.face_encodings(img, known_face_locations=ubicaciones_rostros)
        return ubicaciones_rostros, codificaciones_rostros

    def encontrar_coincidencias(self, codificaciones_rostros):
        coincidencias = []
        distancias = []
        for codificacion in codificaciones_rostros:
            coincidencia = fr.compare_faces(self.rostros_codificados, codificacion, 0.6)
            distancia = fr.face_distance(self.rostros_codificados, codificacion)
            coincidencias.append(coincidencia)
            distancias.append(distancia)
        return coincidencias, distancias

    def obtener_indice_mejor_coincidencia(self, distancias):
        if len(distancias) > 0:
            return np.argmin(distancias)
        return None

    def dibujar_circulo_y_marca_tiempo(self, img, ubicacion_rostro):
        arriba, derecha, abajo, izquierda = ubicacion_rostro
        centro_x, centro_y = (izquierda + derecha) // 2, (arriba + abajo) // 2
        radio = (derecha - izquierda) // 2
        cv2.circle(img, (centro_x, centro_y), radio, (0, 255, 0), 2)

        marca_tiempo = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img, marca_tiempo, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def ejecutar(self):
        img = self.tomar_foto()
        ubicaciones_rostros, codificaciones_rostros = self.reconocer_rostro(img)

        if len(codificaciones_rostros) == 0:
            print("No se encontraron rostros")
            return img, None

        coincidencias, distancias = self.encontrar_coincidencias(codificaciones_rostros)
        mejor_indice_coincidencia = self.obtener_indice_mejor_coincidencia(distancias[0])

        if mejor_indice_coincidencia is not None and distancias[0][mejor_indice_coincidencia] < 0.6:
            nombre = self.nombres_rostros[mejor_indice_coincidencia]
            print(f"Bienvenido {nombre}")
            self.dibujar_circulo_y_marca_tiempo(img, ubicaciones_rostros[0])
            return img, ubicaciones_rostros[0]
        else:
            print("No se encontró coincidencia")
            self.dibujar_circulo_y_marca_tiempo(img, ubicaciones_rostros[0])
            return img, ubicaciones_rostros[0]

class SistemaPrincipal:
    def __init__(self, sr):
        self.sr = sr

    def calcular_area_circulo(self, radio):
        return math.pi * radio ** 2

    def calcular_area_esfera(self, radio):
        return 4 * math.pi * radio ** 2

    def calcular_volumen_esfera(self, radio):
        return (4/3) * math.pi * radio ** 3

    def ejecutar(self):
        img, ubicacion_rostro = self.sr.ejecutar()

        if ubicacion_rostro:
            arriba, derecha, abajo, izquierda = ubicacion_rostro
            radio = (derecha - izquierda) // 2

            area_circulo = self.calcular_area_circulo(radio)
            area_esfera = self.calcular_area_esfera(radio)

            print(f"Área del círculo: {area_circulo:.2f}")
            print(f"Área de la esfera: {area_esfera:.2f}")
        else:
            radio = 10  
            volumen_esfera = self.calcular_volumen_esfera(radio)

            print(f"Volumen de la esfera: {volumen_esfera:.2f}")

        cv2.imshow("Foto del Empleado", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


sr = SistemaReconocimientoFacial()


sistema_principal = SistemaPrincipal(sr)
sistema_principal.ejecutar()

