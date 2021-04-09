import cv2
import numpy as np
from scipy.optimize import curve_fit as fit
import sys

from matplotlib import pyplot as plt

import argparse 
import os

import pdb


drawing = False # true if mouse is pressed
ix = -1 #Creamos un punto inicial x,y
iy = -1,
dotslist = [] #Creamos una lista donde almacenaremos los puntos del contorno

global thick_contour
thick_contour = 2

# mouse callback function
def draw_dots(event,x,y,flags,param): #Crea los puntos de contorno
    
    global ix,iy,drawing, dotslist#Hacemos globales la variabbles dentro de la funcion

    if event == cv2.EVENT_LBUTTONDOWN:#creamos la accion que se realizara si damos click
        drawing = True #Drawinf se vuelve True
        ix = x #Tomamos el punto donde se dio click
        iy = y
        dot = [x,y]
        dotslist.append(dot)#Lo agregamos al dotslist

    elif event == cv2.EVENT_MOUSEMOVE:#Creamos la accion si el mouse se mueve
        if drawing == True: #drawing se vuelve true
            #cv2.circle(img,(x,y),1,(0,0,255),2)
            cv2.line(img, (x,y), (x,y), (0,255,0), thick_contour)#Dibujamos una linea de un solo pixel
            x = x
            y = y
            dot = [x,y]
            dotslist.append(dot)#Agregamos el punto a dotslist
            #print(dotslist) #Imprimimos el dotslist

    elif event == cv2.EVENT_LBUTTONUP:#Cremaos el evento si el boton se levanta
        drawing = False
        #cv2.circle(img,(x,y),1,(0,0,255),1)
        cv2.line(img, (x,y), (x,y), (0,255,0), thick_contour)#Dibujamos la ultima lina en el ultimo punto
      
    return dotslist#Retornamos el dotlist


def Croped(dotslist, img):#hacemos un corte de la imagen en linea recta de tal forma que tenga las 
                          #dimenciones maximas del poligono que creamos
    rect = cv2.boundingRect(dotslist)#Encontramos los limites maximos del
    (x,y,w,h) = rect#Tomamos las dimenciones maximas del dotlist y las guardamos para dimencionar la mascara
    croped = img[y:y+h, x:x+w].copy()#cortamos una seccion rectangular de la imagen
    dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minimo 

    mask = np.zeros(croped.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
    cv2.drawContours(mask, [dotslist2], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    dts = cv2.bitwise_and(croped,croped, mask=mask)#hacemos ceros todos los pixeles externos al contorno, aplicamos la mascara a la imagen
    
    return [dts, mask, croped, dotslist2]

def save_img_with_contour(dotslist, img):
    mask = np.zeros(img.shape[:2], dtype = np.uint8)# creamos una mascara de ceros para poder hacer el corte irregular
    dotslist2 = dotslist- dotslist.min(axis=0)#reajustamos el dotslist con el minim
    cv2.drawContours(mask, [dotslist2], -1, (255,255, 255), -1, cv2.LINE_AA)#dibujamos el contorno
    
    return mask

def histogram(img, mask):
    hist = cv2.calcHist([img], [0], mask, [256], [0,256])
    hist[0] = 0
    hist[255] = 0
    return hist

def Listing(y):
    #Hay que cambiar el codigo para poder usar la funcion flatten
    y1 = []#Creamos una lista vacia
    for i in range(len(y)):#llenamos la lista vacia con los datos de y, esto porque y es de la forma y = [[],[],[]], y necesitamos y = []
        y1.append(y[i][0])  
    
    return y1

def max_values(hist):
    hist = np.asarray(hist)
    max_count = max(hist)
    max_intensity = hist.argmax()
    return [max_count, max_intensity] 

def img_in_memory(file):
    img = cv2.imread(file)#Lee la imagen a color
    return img

def File_Writer(file_name, data_0, data_B, data_G, data_R, fwhm_0, fwhm_B, fwhm_G, fwhm_R):
    
    file = open(str(file_name) + '_Counts_vs_intensity.csv','a')
    file.write('# ' + str(file_name) + '_Counts_vs_intensity'+  '\n')
    
    file.write('# grey max intensity counts value: ' + str(data_0[1]) +  '\n')
    file.write('# grey max intensity value: ' + str(data_0[2]) +  '\n')
    file.write('# Intensity FWHM: ' + str(fwhm_0) + '\n')
    
    file.write('# B max counts value: ' + str(data_B[1]) +  '\n')
    file.write('# B max intensity value: ' + str(data_B[2]) +  '\n')
    file.write('# B FWHM: ' + str(fwhm_B) + '\n')
    
    file.write('# G max counts value: ' + str(data_G[1]) +  '\n')
    file.write('# G max intensity value: ' + str(data_G[2]) +  '\n')
    file.write('# G FWHM: ' + str(fwhm_G) + '\n')
    
    file.write('# R max counts value: ' + str(data_R[1]) +  '\n')
    file.write('# R max intensity value: ' + str(data_R[2]) +  '\n')
    file.write('# R FWHM: ' + str(fwhm_R) + '\n')
    
    file.write('# intensity     counts_intensiy     counts_B     counts_G     counts_R\n')
    
    for h in range(len(data_0[0])):
        file.write(str(h) + '   ' + str(data_0[0][h]) + '    ' + str(data_B[0][h]) + '    ' + str(data_G[0][h]) + '    ' + str(data_R[0][h]) +'\n')
    
    file.close()
    
def MeanAndSigma(x, y):
    mean = sum(x*y)/sum(y) #Calculamos el promedio pesado con los y
    sigma = np.sqrt(sum(y * (x - mean)**2) / (((len(y)-1)/len(y))*sum(y)))#calculamos la desviacion estandar
    #pdb.set_trace()

    return [mean, sigma]

def gaussModel(x, a, x0, sigma):
    f1 = -((x-x0)**2)/(2*(sigma**2))
    y = a*np.exp(f1)

    return y

def gauss(model, x, y, p0):
    popt, pcov = fit(model, x, y, p0)#popt son los parametros del fit, pcov es la curva del fit
    maxg = popt[0]
    meang = popt[1]
    sigmag = popt[2]

    return [maxg, meang, sigmag]

def maximum_value(hist, gauss):
    hist = np.asarray(hist)
    gauss = np.asarray(gauss)
    max_val = max(hist)
    max_ubication = hist.argmax()
    gmax_val = max(gauss)
    gmax_ubication = gauss.argmax()
    
    return [max_val, max_ubication, gmax_val, gmax_ubication]    

def FWHM(sigmag): 
    fwhm = 2*np.sqrt(2*np.log(2))*sigmag
    return fwhm
 
def Fold_Creator(folder_name):
    if os.path.exists(folder_name) == True:#Verificamos que la carpeta no exista
        os.mkdir(folder_name + '_(1)')#Si la carpeta no exite, la creamos.
    elif os.path.exists(folder_name) == False:
        os.mkdir(folder_name)#Si la carpeta no exite, la creamos.
        #fin del for                

def Folder_Change(path):
    try: 
        os.chdir(path)#cambiamos al directorios de los videos
        #fin del try
    except OSError:
        print('La ruta indicada no corresponde a ningun directorio existente: ')
    

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg

file = args['image']
#file = str(sys.argv[1])

#Rfactor= 0.20

img = cv2.imread(file)#Lee la imagen a color
img_0 = cv2.imread(file,cv2.IMREAD_GRAYSCALE)#Lee la imagen pero en intensidad (B and W)
img_B = img[:,:,0]#Separamos al canal B
img_G = img[:,:,1]#Separamos al canal G
img_R = img[:,:,2]#Separamos al canal R


cv2.namedWindow(file, cv2.WINDOW_NORMAL)#Cremaos la ventana para mistras a img
cv2.setMouseCallback(file,draw_dots) #llamamos al MouseCall para dibujar el contorno


name1 = file[0:-4]#definimos un nombre sin la extrension del archivo

#Espacio_de_graficacion_1_de_histograma
fig1 = plt.figure(figsize=(20,15))
ax1 = fig1.add_subplot(111)
plt.ion()

#ax1.set_xlim(0,256)
ax1.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax1.set_xlabel('Intensity Values', fontsize= 12)
ax1.set_ylabel('Counts', fontsize= 12)
ax1.set_title('Histogram of: ' + name1)

#Espacio_de_graficacion_2_de_histograma e imagen
fig2 = plt.figure(figsize=(20,15))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
ax2 = fig2.add_subplot(121)

plt.ion()
ax2.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax2.set_xlabel('Intensity Values', fontsize= 12)
ax2.set_ylabel('Counts', fontsize= 12)
ax2.set_title('Image: ' + name1)

ax3 = fig2.add_subplot(122)
ax3.set_xlabel('x pixel position', fontsize= 12)
ax3.set_ylabel('y pixel position', fontsize= 12)
ax3.set_title('Histogram of: ' + name1)
plt.ion()

plt.show(block=False)

i = 1
while(1):
    cv2.imshow(file,img) #Mostramos a img en la ventana para dibujar el contono

    k = cv2.waitKey(10) & 0xFF
    #l = cv2.waitKey(1) & 0xFF
    
    if k == ord('a'): #space 
        
        print('Corte_aplicado')  
        
        dotslist = np.asarray(dotslist)#Convertimos el contorno en un array de numpy
        
        #Aplicamos el contorno a la image a partir de dtolist
        img_croped_BB_0 = Croped(dotslist, img_0)[0]#Recuperamos solo la region de interes de la imagen de intensidad(imagen cortada con bordes negros=Black Borders)
        img_croped_BB_B = Croped(dotslist, img_B)[0]#Recuperamos solo la region de interes del canal B(imagen cortada con bordes negros=Black Borders)
        img_croped_BB_G = Croped(dotslist, img_G)[0]#Recuperamos solo la region de interes del canal G(imagen cortada con bordes negros=Black Borders)
        img_croped_BB_R = Croped(dotslist, img_R)[0]#Recuperamos solo la region de interes del canal R(imagen cortada con bordes negros=Black Borders)
        
        mask = Croped(dotslist, img_0)[1] #Recuperamos la mascara creada
        
        img_croped_0 = Croped(dotslist, img_0)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara     
        img_croped_B = Croped(dotslist, img_B)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara    
        img_croped_G = Croped(dotslist, img_G)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara    
        img_croped_R = Croped(dotslist, img_R)[2]#recuperamos la imagen cortada en rectangulo para analizar con la mascara    
        
        dotslist = dotslist.tolist()
        
        #Sacamos el histograma
        hist_0 = Listing(histogram(img_croped_0, mask))#Calculamos el histograma usando la mascara en la imagen de intensidad #len(hist) = 256
        hist_B = Listing(histogram(img_croped_B, mask))#Calculamos el histograma usando la mascara en el canal B 
        hist_G = Listing(histogram(img_croped_G, mask))#Calculamos el histograma usando la mascara en el canal G
        hist_R = Listing(histogram(img_croped_R, mask))#Calculamos el histograma usando la mascara en el canal R
        
        #pdb.set_trace()
        #obtencion de los maximos de intensidad y cuentas a partir de los histogramas
        [max_count_0, max_intensity_0] = max_values(hist_0)
        [max_count_B, max_intensity_B] = max_values(hist_B)
        [max_count_G, max_intensity_G] = max_values(hist_G)
        [max_count_R, max_intensity_R] = max_values(hist_R)
        
        #realizamos el ajuste gaussino para obtener el FWHM
        x_0 = np.linspace(0.0, len(hist_0), len(hist_0))#este es el arreglo del eje x para el calculo de ajuste
        
        #obtencion del ajuste gaussiano a partir del histograma de intensida
        [mean_0, sigma_0] = MeanAndSigma(x_0, hist_0)
        [maxg_0, meang_0, sigmag_0] = gauss(gaussModel, x_0, hist_0, p0=[max(hist_0), mean_0, sigma_0])#popt son los parametros del fit u pcov es la curva del fit
        fit_0 = gaussModel(x_0, maxg_0, meang_0, sigma_0)  
        fwhm_0 = FWHM(sigma_0)
        
        #obtencion del ajuste gaussiano a partir del canal B
        [mean_B, sigma_B] = MeanAndSigma(x_0, hist_B)
        [maxg_B, meang_B, sigmag_B] = gauss(gaussModel, x_0, hist_B, p0=[max(hist_B), mean_B, sigma_B])#popt son los parametros del fit u pcov es la curva del fit
        fit_B = gaussModel(x_0, maxg_B, meang_B, sigma_B)  
        fwhm_B = FWHM(sigma_B)
        
        #obtencion del ajuste gaussiano a partir del canal G
        [mean_G, sigma_G] = MeanAndSigma(x_0, hist_G)
        [maxg_G, meang_G, sigmag_G] = gauss(gaussModel, x_0, hist_G, p0=[max(hist_G), mean_G, sigma_G])#popt son los parametros del fit u pcov es la curva del fit
        fit_G = gaussModel(x_0, maxg_G, meang_G, sigma_G)  
        fwhm_G = FWHM(sigma_G)
        
        #obtencion del ajuste gaussiano a partir del canal R
        [mean_R, sigma_R] = MeanAndSigma(x_0, hist_R)
        [maxg_R, meang_R, sigmag_R] = gauss(gaussModel, x_0, hist_R, p0=[max(hist_R), mean_R, sigma_R])#popt son los parametros del fit u pcov es la curva del fit
        fit_R = gaussModel(x_0, maxg_R, meang_R, sigma_R)  
        fwhm_R = FWHM(sigma_R)
        
        #pdb.set_trace()
        #impresion de resultados por terminal
        print('Intesity: max count: ' + str(max_count_0) + '\n' + 'max intensity: ' + str(max_intensity_0) + '\n' + 'FWHM: ' +str(fwhm_0) + '\n')
        print('Channel B: max count: ' + str(max_count_B) + '\n' + 'max intensity: ' + str(max_intensity_B) + '\n' + 'FWHM: ' +str(fwhm_B) + '\n')
        print('Channel G: max count: ' + str(max_count_G) + '\n' + 'max intensity: ' + str(max_intensity_G) + '\n' + 'FWHM: ' +str(fwhm_G) + '\n')
        print('Channel R: max count: ' + str(max_count_R) + '\n' + 'max intensity: ' + str(max_intensity_R) + '\n' + 'FWHM: ' +str(fwhm_R) + '\n')

        #pdb.set_trace()
        
    if k == ord('s'):
        
        print('Datos guardados')
        #Analisis
        #x = np.linspace(0.0, len(hist), len(hist)) #creamos los x para hace el fit, (inicio, final, numero total)
        #pdb.set_trace()

        #Histograma
        #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html
        #ax1.set_xticks(np.arange(0, len(x), step=20))
        line0, = ax1.plot(np.linspace(0, max(hist_0), 256, endpoint=False))
        line0.set_ydata(hist_0)#aqui se asigna al histograma de intensidad como los datos en y que se utilizaran para poder en la grafica
        line0.set_color(color='black')
        line0.set_label('Intensity')
        
        line1, = ax1.plot(np.linspace(0, max(hist_0), 256, endpoint=False))
        line1.set_ydata(fit_0)#aaui se asigna a la curva del fit gaussiano para los datos de intensidad al eje y para poder graficar
        line1.set_color(color='grey')
        line1.set_label('Intensity fit')
        
        line2, = ax1.plot(np.linspace(0, max(hist_B), 256, endpoint=False))
        line2.set_ydata(hist_B)#aqui se asigna al histograma de intensidad como los datos en y que se utilizaran para poder en la grafica
        line2.set_color(color='blue')
        line2.set_label('Blue channel')
        
        #line3, = ax1.plot(np.linspace(0, max(hist_B), 256, endpoint=False))
        #line3.set_ydata(fit_B)#aaui se asigna a la curva del fit gaussiano para los datos de intensidad al eje y para poder graficar
        #line3.set_color(color='slateblue')
        #line3.set_label('Blue channel fit')
        
        line4, = ax1.plot(np.linspace(0, max(hist_G), 256, endpoint=False))
        line4.set_ydata(hist_G)#aqui se asigna al histograma de intensidad como los datos en y que se utilizaran para poder en la grafica
        line4.set_color(color='green')
        line4.set_label('Green Channel')
        
        #line5, = ax1.plot(np.linspace(0, max(hist_G), 256, endpoint=False))
        #line5.set_ydata(fit_G)#aaui se asigna a la curva del fit gaussiano para los datos de intensidad al eje y para poder graficar 
        #line5.set_color(color='lime')
        #line5.set_label('Green Channel fit')
        
        line6, = ax1.plot(np.linspace(0, max(hist_R), 256, endpoint=False))
        line6.set_ydata(hist_R)#aqui se asigna al histograma de intensidad como los datos en y que se utilizaran para poder en la grafica
        line6.set_color(color='red')
        line6.set_label('Red Channel')
         
        #line7, = ax1.plot(np.linspace(0, max(hist_R), 256, endpoint=False))
        #line7.set_ydata(fit_R)#aaui se asigna a la curva del fit gaussiano para los datos de intensidad al eje y para poder graficar
        #line7.set_color(color='lightcoral')
        #line7.set_label('Red Channel fit')
        
        ax1.set_xlabel('Intensity Values', fontsize= 12)
        ax1.set_ylabel('Counts', fontsize= 12)
        fig1.canvas.draw()#updatea el grafico con la nueva curva
        ax1.draw_artist(ax1.patch)#selecciona al arreglo de ax, para darle rango al eje x
        
        ax1.draw_artist(line0) #Redibuja line1 solo si es necesario
        ax1.draw_artist(line1) #Redibuja linew2 solo si es necesario
        ax1.draw_artist(line2) #Redibuja line1 solo si es necesario
        #ax1.draw_artist(line3) #Redibuja linew2 solo si es necesario
        ax1.draw_artist(line4) #Redibuja line1 solo si es necesario
        #ax1.draw_artist(line5) #Redibuja linew2 solo si es necesario
        ax1.draw_artist(line6) #Redibuja linew2 solo si es necesario
        #ax1.draw_artist(line7) #Redibuja linew2 solo si es necesario

        
        ax1.text(120, 120, 'max count: ' + str(max_count_0) + '\n' + 'max intensity: ' + str(max_intensity_0), fontsize=12)
        ax1.legend(loc='best')
        
        
        #Imagen e histograma
        ax3.set_xlabel('x pixel position', fontsize= 12)
        ax3.set_ylabel('y pixel position', fontsize= 12)
        ax3.imshow(img_croped_BB_0, cmap= 'Greys')
        
        
        line8, = ax2.plot(np.linspace(0, max(hist_0), 256, endpoint=False))
        line8.set_ydata(hist_0)
        line8.set_color(color='black')
        line8.set_label('Intensity')
        
        line9, = ax1.plot(np.linspace(0, max(hist_0), 256, endpoint=False))
        line9.set_ydata(fit_0)
        line9.set_color(color='black')
        line9.set_label('Intensity fit')
        
        fig2.canvas.draw()#updatea el grafico con la nueva curva
        ax2.set_xlabel('Intensity Values', fontsize= 12)
        ax2.set_ylabel('Counts', fontsize= 12)
        ax2.draw_artist(ax2.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax2.draw_artist(line2) #Redibuja line solo si es necesario
        
        
        Fold_Creator('Results_Data_Analisis_Image:_'+ name1 + '_' + str(i))
        Folder_Change('Results_Data_Analisis_Image:_'+ name1 + '_' + str(i))
        
        fig1.canvas.flush_events()#Hace un sleep para que se pueda crear la grafic
        fig1.savefig('Histogram of ' + name1 +'_' +str(i) + '.svg')   
        fig2.canvas.flush_events()#Hace un sleep para que se pueda crear la grafica
        fig2.savefig('Image_and_histogram_of_' + name1 + '_' +str(i) + '.svg')
        
        #plt.show()
        cv2.imshow('croped', img_croped_BB_0)#Mostramos img2 con el contorno 
        cv2.imwrite("Corte_" + name1 + '_' + str(i) +'_.jpg', img_croped_BB_0) 
        
        File_Writer('Data_Analisis_Image:_'+ name1 + '_' + str(i), 
                    [hist_0, max_count_0, max_intensity_0], 
                    [hist_B, max_count_B, max_intensity_B],
                    [hist_G, max_count_G, max_intensity_G], 
                    [hist_R, max_count_R, max_intensity_R], 
                    fwhm_0,
                    fwhm_B,
                    fwhm_G,
                    fwhm_R
                    )
        
        selected_area = cv2.drawContours(img, [np.asarray(dotslist)], -1, (0,255,0), thick_contour)#Dibujamos el contorno
        cv2.imwrite("Area_seleccionada_numero_" + str(i) + "de:" + name1 +'_.jpeg', selected_area)
        
        Folder_Change('..')
        
        i += 1#Para poder guardar en la siguiente ejecucion sin sobreescribir la anterior

    if k == ord('d'):
        print('Contorno y datos borrados')
        hist_0.clear()
        dotslist.clear()
        ax1.clear()
        ax2.clear()
        #cv2.destroyAllWindows()#Destruimos todas las ventanas
        cv2.namedWindow(file, cv2.WINDOW_NORMAL)
        img = img_in_memory(file)
        fwhm_0 = 0
        fwhm_B = 0
        fwhm_G = 0
        fwhm_R = 0
        
    if k == ord('q'):#esc
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc

cv2.destroyAllWindows()#Destruimos todas las ventanas
