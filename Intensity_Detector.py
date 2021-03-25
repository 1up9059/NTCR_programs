import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
import pdb


def nothing(x):
    pass
#--------------------------------------------------------------------------------------------------------------------------------------------------
def nonzero(img, lum, b, g, r):
    
    countimg = cv2.countNonZero(img)
    countlum = cv2.countNonZero(lum)
    countb = cv2.countNonZero(b)
    countg = cv2.countNonZero(g)
    countr = cv2.countNonZero(r)
    
    return [countimg, countlum, countb,countg,countr]
#--------------------------------------------------------------------------------------------------------------------------------------------------
def DivIMG(img):#Dvidimos la imagen en HSV
    
    #Dividimos la imagen en B, G, R
    b = img[:,:,0]#Separacion de canal b
    g = img[:,:,1]#Separacion de canal g
    r = img[:,:,2]#Separacion de canal r
    
    return [b, g, r]
#--------------------------------------------------------------------------------------------------------------------------------------------------
def Black_and_White(b, g, r):   
    #Pasamos la imagen a blanco y negro 
    holder = np.ones(b[:,:,0].shape)
    for i in range(b.shape[0]):#Filas
        for j in range(b.shape[1]):#Columnas
            lum=0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j] # Tipo Photometric/digital ITU BT.709
            #lum=0.299*r[i][j] +0.587*g[i][j]+0.114*b[i][j] #Tipo Digital ITU BT.601 (gives more weight to the R and B components)
            holder[i][j] = lum
    
    return holder
#--------------------------------------------------------------------------------------------------------------------------------------------------
def listing(entry_list):#cambia el formato de la lista de [[],[],[],[],[],[],[],[],[]] a [,,,,,,]
    
    final_list = []
    for i in range(len(entry_list)):
        final_list.append(entry_list[i][0])
        
    final_list[0] = 0
    
    return final_list
#-------------------------------------------------------------------------------------------------------------------------------------------------
def sumation(entry_list):#Suma todas las cuentas del histograma
    count = 0
    for i in entry_list:
        count += i
    
    return count 
#------------------------------------------------------------------------------------------------------------------------------------------------
def comparison(sum_img1, sum_img2):#Hace la comparacion por porcentaje de todas la imagen seccionada con respecto a la imagen total
    
    if sum_img2 != 0:
        percent = (sum_img1/sum_img2)*100
    elif sum_img2 == 0:
        percent = 0
        
    return percent
#------------------------------------------------------------------------------------------------------------------------------------------------
def File_Writer(file_name, data1):#Escribe el archivo que tiene los datos de un histograma
    file = open(str(file_name) + '_Counts_vs_intensity.csv','a')
    for h in range(len(data1)):
        file.write(str(h) + '	'+ str(data1[h]) +'\n')
    file.close()

###################################################################################################################################################              
           
#creamos la ventana de los controladores
cv2.namedWindow('image', cv2.WINDOW_NORMAL)#Crea la ventana de los controladores inicia

#CREACION DE CONTROLADORES
cv2.createTrackbar('Low Intensity value','image',0, 255, nothing)#Crea el controlador de valor alto
cv2.createTrackbar('High Intensity value','image', 0, 255, nothing)#Crea el controlador de valor bajo

#LECTURA DE LA IMAGEN  
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg

#Redimencionamos la imagen
img_1 = cv2.imread(args['image'])#Lee la imagen en BGR
img_2 = img_1.copy()#Crea una copia de la imgen, con esta se trabaja
img_G_1 = cv2.cvtColor(img_1.copy(), cv2.COLOR_BGR2GRAY)#Hacemos otra copia de la imagen pero en blaco y negro
img_G_2 = img_G_1.copy()#Copiamos la imagen en blanco y negro
#img_HSV_1 = cv2.cvtColor(img_1.copy(), cv2.COLOR_BGR2HSV) #Hacemos una copia de la imagen original pero en HSV

#cv2.namedWindow('img', cv2.WINDOW_NORMAL)#Creamos la imagen que muestra a 
#cv2.namedWindow('image2', cv2.WINDOW_NORMAL)#Creamos la ventana para mistras a img
#cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
#cv2.namedWindow('image4', cv2.WINDOW_NORMAL)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.ion()
plt.autoscale(enable=True, tight = True)

hist_img_1 = listing(cv2.calcHist(img_G_2, [0], None, [256], [0,256]))
maximum = max(hist_img_1) 
File_Writer("Histogram_Data_" + args['image'] + '_complete' + '_in_gray' , hist_img_1)

line_img_G_2, = ax1.plot(np.linspace(0, maximum, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
line_output_G, = ax1.plot(np.linspace(0, maximum, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
#pdb.set_trace()

#ax1.set_xlim(0,256)#limite de numeracion del eje x
#ax1.set_ylim([0, 1.1])#Limite de eje y
ax1.set_title('Histograms')
ax1.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax1.set_xlabel('Intensity', color = 'black')
ax1.set_ylabel('Counts', color = 'black')

#plt.show(block=False)
fig1.canvas.draw()

i = 1

###################################################################################################################################################

while(1):
    
    #cv2.imshow('img', img_1)#Mostramos la imagen original
    k = cv2.waitKey(10) & 0xFF#Creamos el boton de interaccion
    
    #salida del programa-----
    if k == ord('q'):
        break
  
    #Tratamiento de imagen------------------------------------------------------------------------------------------------------
    
    low_I = cv2.getTrackbarPos('Low Intensity value', 'image')#Cremos el controlador de valor bajo
    high_I = cv2.getTrackbarPos('High Intensity value', 'image')#Creamos el controlador de valor alto
    
    lower_I = np.array([low_I], dtype = "uint8")#Controlador de valores bajo en el loop
    higher_I = np.array([high_I], dtype = "uint8")#Controlador de valores alto en el loop 
    
    mask = cv2.inRange(img_G_1, lower_I, higher_I)#Creamos la mascara bajo el rango seleccionado de valores de intensidad
    output = cv2.bitwise_and(img_2, img_2, mask = mask)#Aplicamos la mascara a img_2, y separamos lo seleccionado
    output_G = cv2.bitwise_and(img_G_2, img_G_2, mask = mask)#Aplicamos la mascara al img_G_2 y separamos lo seleccionado
    #ret1, thresh = cv2.threshold(mask, 127,225,0)#Hacemos el treshhold para la creacion del contorno
    #contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#Creamos el contorno para la region
    #img_2_WC = cv2.drawContours(img_2, contours, -1, (0,255,0), 1)#Dibujamos el contorno
    
    #Mostramos las imagenes
    #cv2.imshow('image2', mask)#Mostramos la mascara usada
    #cv2.imshow('image3', output)#Mostramos la imagen seccionada pero a color
    #cv2.imshow('image4', output_G) #Mostramos la imagen seccionada en blanco y negor
    
    #Analisis-------------------------------------------------------------------------------------------------------------------
    if np.count_nonzero(output) and np.count_nonzero(output_G) != 0:#Nos fijamos que en ambos casos el output y el output_G no sean solo ceros
        
        #Calculamos el histograma
        hist_img_G_2 = listing(cv2.calcHist(img_G_2, [0], None, [256], [0,256])) #Calculamos el histograma de img_G_2, sin mascara
        hist_output_G = listing(cv2.calcHist(output_G, [0], None, [256], [0,256])) #Calculamos el histograma de hist_output_G sin mascara
                                                                                   #Porque ya separamos las areas de interes 
        #pdb.set_trace()
        
        #Graficacion Asignamos los datos a las lineas
        line_img_G_2.set_ydata(hist_img_G_2)#Asignamos la curva del histograma de hist_img_G_2 a line_img_G_2
        line_output_G.set_ydata(hist_output_G)#Asignamos la curva del hist_output_G a line_output_G
        
        #Creamos las etiquetas
        line_img_G_2.set_label('Luminosity of Image in BW')#Ponemos la etiqueta a line_img_G_2
        line_output_G.set_label('Luminisity of sections')#Ponemos la etiqueta a line_output_G
        ax1.legend(loc='best')#hacemos que las etiquetas se pongan en el mejor lugar
        #pdb.set_trace()
        
        ax1.draw_artist(ax1.patch)#Dibujamos el espacio ax1
        ax1.draw_artist(line_img_G_2)#dibujamos la linea line_img_G_2 
        ax1.draw_artist(line_output_G)#dibujamos la linea line_output_G
        fig1.canvas.draw()#updatea el grafico con la nueva curva
        fig1.canvas.flush_events()#Refrescamos el canvas
    
        #Sumamos el histograma
        sum_img_G_2 = sumation(hist_img_G_2)
        sum_output_G = sumation(hist_output_G)
        #pdb.set_trace()
        
    
    if k == ord('s'):#Guardamos los datos--------------------------------------------------------------------------------------------
        

        percent = comparison(sum_output_G, sum_img_G_2)
        print('Porcentaje de imagen: ' + str(percent)) 
        print('Datos guardados')
        ax1.text(120, 120,'Porcentaje: ' + str(str(percent)))
        cv2.imwrite("Seccion" + args['image'] + '_' + str(i) +'_.jpg', output) 
        cv2.imwrite("Seccion" + args['image'] + '_in_gray' + '_' + str(i) +'_.jpg', output_G) 
        File_Writer("Histogram_Data_" + args['image'] + '_in_gray' + '_' + str(i) +'_' , hist_output_G)
        fig1.savefig('Image_and_histogram_of_section:' + args['image'] + '_' +str(i) + '.png')
        
        i += 1




cv2.destroyAllWindows()#Destruimos todas las ventanas