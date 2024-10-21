import cv2 as cv2
import face_recognition as fr
import os
from pathlib import Path


def foto_one():
    #crear base de datos
    #carpeta = Path("Empleados")
    carpeta = Path("fotos nays")


    #carga de fotos 
    img_oneone = []
    for ruta in carpeta.iterdir():
        try:
            img_empleado = fr.load_image_file(ruta)
            img_empleado = cv2.cvtColor(img_empleado,cv2.COLOR_BGR2RGB)
            img_oneone.append(img_empleado)
        except Exception as e:
            continue
        
    print(len(img_oneone))

    # localizar cara control 
    ubi_cara = []
    cod_cara = []

    for cara in img_oneone:
        #localizar caras 
        lugar_cara = fr.face_locations(cara)[0]
        #codifica las caras como una independiente efirma
        cara_codificada = fr.face_encodings(cara)[0]
        ubi_cara.append(lugar_cara)
        cod_cara.append(cara_codificada)
    return ubi_cara,cod_cara


def fotos_prueba(ubi_cara,code_one):
    #crear base de datos
    #carpeta = Path("Empleados")
    carpeta = Path("fotos_prueba")


    #carga de fotos 
    img_test = []
    for ruta in carpeta.iterdir():
        try:
            img_empleado = fr.load_image_file(ruta)
            #print(img_empleado)
            img_empleado = cv2.cvtColor(img_empleado,cv2.COLOR_BGR2RGB)
            img_test.append(img_empleado)
        except Exception as e:
            continue
        

    # localizar cara control 
    ubi_cara = []
    cod_cara = []

    for cara in img_test:
        #localizar caras 
        lugar_cara = fr.face_locations(cara)[0]
        #codifica las caras como una independiente efirma
        cara_codificada = fr.face_encodings(cara)[0]
        ubi_cara.append(lugar_cara)
        cod_cara.append(cara_codificada)


    #detectar el rosto con un cuadro 
    for cuadro_id in range(len(img_test)):
        cv2.rectangle(img_test[cuadro_id],
                (ubi_cara[cuadro_id][3],ubi_cara[cuadro_id][0]),
                (ubi_cara[cuadro_id][1],ubi_cara[cuadro_id][2]),
                (0,255,0),
                2)
 
    for i in range(len(cod_cara)):
        resultado = fr.compare_faces(code_one, cod_cara[i],tolerance=0.6)
        if resultado[0] == True:
            cv2.putText(img_test[i],
                f"Es una one one",
                (50,50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,255,0),
                2)
        else:
            cv2.putText(img_test[i],
                f"No es una one one",
                (50,50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0,255,0),
                2)
    



    #mostras imagenes
    for img_id in range(len(img_test)):
        cv2.imshow(f"Foto-{img_id}",img_test[img_id])

    
    
    
ubi_cara,code_one = foto_one()
fotos_prueba(ubi_cara,code_one)

cv2.waitKey(0)

