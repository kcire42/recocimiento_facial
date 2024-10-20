import cv2 as cv2
import face_recognition as fr


#cargar imagenes 
foto_control = fr.load_image_file("img/FotoA.jpg")
foto_prueba = fr.load_image_file("img/FotoB.jpg")

#cambiar estructura de la imagenes

foto_control = cv2.cvtColor(foto_control,cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba,cv2.COLOR_BGR2RGB)

# localizar cara control A
lugar_cara_A = fr.face_locations(foto_control)[0]
#codifica las caras como una independiente efirma
cara_codificada_A = fr.face_encodings(foto_control)[0]

# localizar cara control B
lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

#mostrar rectangulo 
cv2.rectangle(foto_control,
              (lugar_cara_A[3],lugar_cara_A[0]),
              (lugar_cara_A[1],lugar_cara_A[2]),
              (0,255,0),
              2)

#mostrar rectangulo 
cv2.rectangle(foto_prueba,
              (lugar_cara_B[3],lugar_cara_B[0]),
              (lugar_cara_B[1],lugar_cara_B[2]),
              (0,255,0),
              2)

# realizar comparacion 
# con tolareancia ajusto que tanta precision quiero
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B,tolerance=0.6)

# medida de la distancia 

distancia = fr.face_distance([cara_codificada_A],cara_codificada_B)
print(distancia)

#resultado de analizis 
if resultado[0] == True:
    print("Son la misma persona")
else:
    print("Son diferente persona")

# mostrar resultado 
cv2.putText(foto_prueba,
            f"{resultado} {distancia.round(2)}",
            (50,50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,255,0),
            2)

# mostrar imagenes

cv2.imshow("Foto Control",foto_control)
cv2.imshow("Foto Prueba",foto_prueba)



#mantener programa abiert

cv2.waitKey(0)