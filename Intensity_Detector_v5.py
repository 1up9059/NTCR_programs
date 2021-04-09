'''
En este codigo, las variables declaradas cuyo nombre tiene un _ al final, corresponden a variables que estaran variando conforme el programa se ejecuta, mientras que las varibles que no lo tienen, son aquellas que solo son calculadas una vez y luego de eso permanecen iguales siempre. Ademas, las variables que contienen un  C_ al inicio, son aquellas variables que tienen que ver con el analisis conjunto de los canales BGRA de la imagen usada. Mientras que las variables que contienen E_ al inicio del nombre son aquellas que corresponden al analisis separado por cada canala de la imagen usada por el usuario
'''


import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as fit
from scipy import exp, pi
import os
import pdb

def nothing(x):
    pass

def nonzero(img, lum, H, S,V):
    
    countimg = cv2.countNonZero(img)
    countlum = cv2.countNonZero(lum)
    countH = cv2.countNonZero(H)
    countS = cv2.countNonZero(S)
    countV = cv2.countNonZero(V)
    
    return [countimg, countlum, countH,countS,countV]

def DivIMG(img):#Dvidimos la imagen en HSV
    B = img[:,:,0]#Separacion de canal H
    G = img[:,:,1]#Separacion de canal S
    R = img[:,:,2]#Separacion de canal V

    return [B, G, R]

def ReconIMG( b, g, r, gray): 
    #en el zeros es necesario especificar el dtype porque de lo contrario opencv crashea
    img_BGRA = np.zeros((img_BGR.shape[0], img_BGR.shape[1], 4), dtype=np.uint8)#este es le holder que lo que hace es formar un array para formar la imagen
    img_BGRA[:,:,0] = b
    img_BGRA[:,:,1] = g
    img_BGRA[:,:,2] = r
    img_BGRA[:,:,3] = gray
    return img_BGRA

def hist(img_B, img_G, img_R, img_A, mask=None):
    #Calculamos los histogramas de cada canal y el de la intensidad total para graficacion
    
    h_B = cv2.calcHist([img_B], [0], mask, [256], [0,256])#Lo mismo del anterio pero solo al canal H
    h_G = cv2.calcHist([img_G], [0], mask, [256], [0,256])#Lo mismo del anterio pero solo al canal S
    h_R = cv2.calcHist([img_R], [0], mask, [256], [0,256])#Lo mismo del anterio pero solo al canal v
    h_A = cv2.calcHist([img_A], [0], mask, [256], [0,256])#Calculamos el histograma de lum aplicando la mascara (Lum es la imagen pero aplicando la mascara)
  
  #Esta parte no hace falta, pasa el formato de la lista de [[],[],[],[],[],[]...] a [,,,,,,]  
    for i in range(len(h_B)):
        
        h_B[i] = h_B[i][0]
        h_G[i] = h_G[i][0]
        h_R[i] = h_R[i][0]
        h_A[i] = h_A[i][0]

    #Elimiamos la cuenta de valor de intencidad cero, ya que es un pico que no queremos
    h_B[0] = 0
    h_G[0] = 0
    h_R[0] = 0
    h_A[0] = 0

    #pdb.set_trace()
    return [h_B, h_G, h_R, h_A]

def indentify_max(hist_B, hist_G, hist_R):
    maximus = np.asarray([max(hist_B), max(hist_G), max(hist_R)])
    max_count = max(maximus)
    max_intensity = np.asarray([hist_B.argmax(), hist_G.argmax(), hist_R.argmax()])
    return [max_count, maximus, max_intensity]


def File_Writer(file_name, maxs_hists_counts, maxs_hists_inten, hists, FWHM_hists):
    
    [hist_B, hist_G, hist_R, hist_A] = hists
    
    hist_B = hist_B.flatten()
    hist_G = hist_G.flatten()
    hist_R = hist_R.flatten()
    hist_A = hist_A.flatten()
    
    #pdb.set_trace()
    file = open(str(file_name) + '.csv','a')
    file.write('# ' + str(file_name) +  '\n')
    
    file.write('# B max counts value: ' + str(maxs_hists_counts[0]) +  '\n')
    file.write('# B max intensity value: ' + str(maxs_hists_inten[0]) +  '\n')
    file.write('# B FWHM: ' + str(FWHM_hists[0]) + '\n')
    
    file.write('# B max counts value: ' + str(maxs_hists_counts[1]) +  '\n')
    file.write('# B max intensity value: ' + str(maxs_hists_inten[1]) +  '\n')
    file.write('# B FWHM: ' + str(FWHM_hists[1]) + '\n')
    
    file.write('# B max counts value: ' + str(maxs_hists_counts[2]) +  '\n')
    file.write('# B max intensity value: ' + str(maxs_hists_inten[2]) +  '\n')
    file.write('# B FWHM: ' + str(FWHM_hists[2]) + '\n')
    
    file.write('# B max counts value: ' + str(maxs_hists_counts[3]) +  '\n')
    file.write('# B max intensity value: ' + str(maxs_hists_inten[3]) +  '\n')
    file.write('# B FWHM: ' + str(FWHM_hists[3]) + '\n')
    

    
    file.write('# intensity     counts_A     counts_B     counts_G     counts_R\n')
    
    for h in range(len(hist_B)):
        file.write(str(h) + '   ' + str(hist_B[h]) + '    ' + str(hist_G[h]) + '    ' + str(hist_R[h]) + '    ' + str(hist_A[h]) +'\n')
    
    file.close()
    
    
def MeanAndSigma(x, y):
    
    avg_num = sum([j*i for i,j in zip(y,x)])
    avg_dem = sum(x)
    
    avg = avg_num/avg_dem
    
    sd_num = sum([(j*(i-avg)**2) for i,j in zip(y,x)])
    sd_dem = ((len(x)-1)/len(x))*sum(x)
    
    sigma = np.sqrt(sd_num / sd_dem)
    
    #pdb.set_trace()
 
    
    
    #mean = sum(x*y)/sum(y) #Calculamos el promedio pesado con los y
    #sigma = np.sqrt(sum(y * (x - mean)**2) / (((len(y)-1)/len(y))*sum(y)))#calculamos la desviacion estandar


    return [avg, sigma]


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
        
def FWHM(sigmag): 
    fwhm = 2*np.sqrt(2*np.log(2))*sigmag
    return fwhm


#VENTANA DE CONTROLADORES Y LAS VENTANAS DONDE SE MANIPULARAN LAS IMAGENES   
cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray scale selection', cv2.WINDOW_NORMAL)#Creamos la ventana para mistras a img
cv2.namedWindow('BGRA scale selection', cv2.WINDOW_NORMAL)#Creamos la ventana para mistras a img

#CREACION DE CONTROLADORES
#Low red and high blue
cv2.createTrackbar('Low Blue','Original image',0, 255, nothing)
cv2.createTrackbar('High Blue','Original image', 0, 255, nothing)
#Low red and high green
cv2.createTrackbar('Low Green','Original image',0, 255, nothing)
cv2.createTrackbar('High Green','Original image',0,255, nothing)
#Low red and high red
cv2.createTrackbar('Low Red','Original image', 0,255, nothing)
cv2.createTrackbar('High Red','Original image', 0,255, nothing)

#Low red and high grey
cv2.createTrackbar('Low Gray','Original image', 0,255, nothing)
cv2.createTrackbar('High Gray','Original image', 0,255, nothing)
#---------------------------------------------------------------
    
#LECTURA DE LA IMAGEN  
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to the image')
args = vars(ap.parse_args())
#llamamos asi #python Color_Detector_1.py --image 1.jpg
file_name = args['image']

#---------------------------------------------------------------

#LEEMOS LA IMAGEN
img_BGR = cv2.imread(args['image'])

#CREAMOS LA IMAGEN A CUATRO CANALES Y UNA COPIA PARA PODER TRABAJARLA
img_A = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY) #pasamos la imagen a grey tambien
[img_B, img_G, img_R] = DivIMG(img_BGR)#separamos los canales b g r
img_BGRA = ReconIMG(img_B, img_G, img_R, img_A) #hacemos una reconvinacion para hacer la image RGBA
img_BGRA_ = img_BGRA.copy()

#-------------------------------------------------------------------
'''Aca creamos los espacios de graficacion para el analisis conjunto de todos los canales BGRA'''

#CREACION DEL ESPACIO DE GRAFICACION 1, ESTE ES PARA EL HISTOGRAMA BGR
fig1 = plt.figure(figsize=(20,15))
gs1 = fig1.add_gridspec(1, 2, hspace=0, wspace=0.176)
(ax1, ax2) = gs1.subplots()
plt.ion()

C_line_B, = ax1.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
C_line_G, = ax1.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
C_line_R, = ax1.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio

C_line_B_, = ax1.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
C_line_G_, = ax1.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
C_line_R_, = ax1.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio

C_line_B_.set_label('red_section')
C_line_G_.set_label('green_section')
C_line_R_.set_label('blue_section')
        
C_line_B_.set_color('dodgerblue')
C_line_G_.set_color('lime')
C_line_R_.set_color('indianred')

C_line_B.set_color('blue')
C_line_G.set_color('green')
C_line_R.set_color('red')

ax1.set_xlim(0,256)#limite de numeracion del eje x
ax1.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax1.set_title('Histograms BGR Scale')
ax1.set_xlabel('Intensity', color = 'black')
ax1.set_ylabel('Counts', color = 'black')

#CREACION DEL ESPACIO DE GRAFICACION 2. ESTE ES PARA LOS HISTOGRAMAS BGR 
plt.ion()
plt.autoscale(enable=True, tight = True)

C_line_A, = ax2.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
C_line_A_, = ax2.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio

C_line_A_.set_color('gray')
C_line_A_.set_label('Gray section')
C_line_A.set_color('black')
C_line_A.set_label('Gray histogram')

ax2.set_xlim(0,256)#limite de numeracion del eje x
ax2.set_title('Histograms Gray')
ax2.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax2.set_xlabel('Intensity', color = 'black')
ax2.set_ylabel('Counts', color = 'black')

plt.show(block=False)
fig1.canvas.draw()


#---------------------------------------------------------------------------------
'''En esta seccion crearemos los espacios de graficacion para mostrar los histogramas de analisis por cada canal separado
Esto incluye los las curvas individualizadas para cada curva para los canales B, G, R y A'''


#CREACION DEL ESPACIO DE GRAFICACION 3, ESTE ES PARA EL HISTOGRAMA BGR
fig2 = plt.figure(figsize=(20,15))
gs2 = fig2.add_gridspec(1, 2, hspace=0, wspace=0.17)
(ax3, ax4) = gs2.subplots()

plt.ion()

E_line_B, = ax3.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
E_line_G, = ax3.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
E_line_R, = ax3.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio

E_line_B_, = ax3.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
E_line_G_, = ax3.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
E_line_R_, = ax3.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio

E_line_B_.set_label('red_section')
E_line_G_.set_label('green_section')
E_line_R_.set_label('blue_section')
        
E_line_B_.set_color('dodgerblue')
E_line_G_.set_color('lime')
E_line_R_.set_color('indianred')

E_line_B.set_color('blue')
E_line_G.set_color('green')
E_line_R.set_color('red')

ax3.set_xlim(0,256)#limite de numeracion del eje x
ax3.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax3.set_title('Histograms BGR Scale')
ax3.set_xlabel('Intensity', color = 'black')
ax3.set_ylabel('Counts', color = 'black')

#CREACION DEL ESPACIO DE GRAFICACION 4. ESTE ES PARA LOS HISTOGRAMAS BGR

E_line_A, = ax4.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio
E_line_A_, = ax4.plot(np.linspace(0, 256, 256, endpoint=True))#Valores para el eje X, en este caso ponemos un arreglo vacio

E_line_A_.set_color('gray')
E_line_A_.set_label('Gray section')
E_line_A.set_color('black')
E_line_A.set_label('Gray histogram')

ax4.set_xlim(0,256)#limite de numeracion del eje x
ax4.set_title('Histograms Gray')
ax4.set_xticks(np.linspace(0, 256, 10))#ajusta las etiquetas en x
ax4.set_xlabel('Intensity', color = 'black')
ax4.set_ylabel('Counts', color = 'black')

fig1.canvas.draw()
fig2.canvas.draw()
plt.show(block=False)

#---------------------------------------------------------------------------------
i = 0

#LOOP DE EJECUCION
while(1):
    cv2.imshow('Original image', img_BGR)
    k = cv2.waitKey(10) & 0xFF
    
    
    if k == ord('q'):#exit
        #pdb.set_trace()
        break# hace,el break para para el programa si se estripa esc
    
    #CONTROLADORES DURANTE LA EJECUCION DE
    low_B = cv2.getTrackbarPos('Low Blue', 'Original image')
    high_B = cv2.getTrackbarPos('High Blue', 'Original image')
    low_G = cv2.getTrackbarPos('Low Green', 'Original image')
    high_G = cv2.getTrackbarPos('High Green', 'Original image')
    low_R = cv2.getTrackbarPos('Low Red', 'Original image')
    high_R = cv2.getTrackbarPos('High Red', 'Original image')
    low_A = cv2.getTrackbarPos('Low Gray', 'Original image')
    high_A = cv2.getTrackbarPos('High Gray', 'Original image')
    #Arrays de los valores de los controladores
    lower_BGRA = np.array([low_B, low_G, low_R, low_A], dtype = "uint8")
    higher_BGRA = np.array([high_B, high_G, high_R, high_A], dtype = "uint8")
    #lower_BGRA = np.array([low_G, low_R, low_A, low_B], dtype = "uint8")
    #higher_BGRA = np.array([high_G, high_R, high_A, high_B], dtype = "uint8")
    
    '''
    La siguiente mascara se usa para hacer una seleccion basada en todos los canales en conjunto. Lo cual quiere decir que en la imagen, cada pixel es un punto de coordenadas BGRA, asi, la mascara de seleccion toma 4 valores, es decir, que si un pixel RGBA no esta en rango al mens uno de los canales, esto hara que el pixel quede excluido y no sea considerado para la generacion de la imagen output_RGBA_
    '''
    mask_BGRA = cv2.inRange(img_BGRA, lower_BGRA, higher_BGRA)
  
    '''
    Las siguientes mascaras se utilizan para hacer una seleccion basada los canales BGRA individualmente. Lo que quiere decir que cada mascara trabaja individualemente con cada canal de tal forma que no importa si un pixel con coordenadas BGRA tiene un valor en uno de sus canales que no esta en rango, igualemente este sera tomado en cuenta en la seleccion de los pixeles de los demas canales para las la generacion de las imagenes output_B_, output_G_, output_R_, output_A_
    '''
    mask_B = cv2.inRange(img_BGRA[:,:,0], low_B, high_B)
    mask_G = cv2.inRange(img_BGRA[:,:,1], low_G, high_G)
    mask_R = cv2.inRange(img_BGRA[:,:,2], low_R, high_R)
    mask_A = cv2.inRange(img_BGRA[:,:,3], low_A, high_A)
    
    '''output_BGRA contiene a todos los pixeles que tienen todos sus valores dentro de los rangos seleccionados para el analsis'''
    C_output_BGRA_ = cv2.bitwise_and(img_BGRA, img_BGRA, mask = mask_BGRA)#Se la palicamos a img la imagen original
    
    '''los output_B_, los output_G_, los output_R_, los output_A_ , son las imagenes que contienen los pixles que estan en rango para cada canala de la imagen de manera individual
    '''
    E_output_B_ = cv2.bitwise_and(img_BGRA[:,:,0], img_BGRA[:,:,0], mask = mask_B)#Se la palicamos a img la imagen original
    E_output_G_ = cv2.bitwise_and(img_BGRA[:,:,1], img_BGRA[:,:,1], mask = mask_G)#Se la palicamos a img la imagen original
    E_output_R_ = cv2.bitwise_and(img_BGRA[:,:,2], img_BGRA[:,:,2], mask = mask_R)#Se la palicamos a img la imagen original
    E_output_A_ = cv2.bitwise_and(img_BGRA[:,:,3], img_BGRA[:,:,3], mask = mask_A)#Se la palicamos a img la imagen original
    
 
    #ANALISIS DE IMAGEN OUTPUT
    if np.count_nonzero(C_output_BGRA_) != 0:#EStablecemos la condicion para asegurar que los histogramas solo se calculen cuando hay pixeles en rango

        #USAMOS LAS  IMAGENES DE 4 CANALAES PARA SACAR LOS HISTOGRAMAS
        '''Estos histogramas corresponden a la imagen BGRA completa'''
        [hist_B, hist_G, hist_R, hist_A] = hist(img_BGRA[:,:,0], img_BGRA[:,:,1], img_BGRA[:,:,2], img_BGRA[:,:,3])
        
        '''Estos histogramas corresponde al analsis conjunto excluyente de la imagen vbasado en mask_BGRA'''
        [C_hist_B_, C_hist_G_, C_hist_R_, C_hist_A_] = hist(C_output_BGRA_[:,:,0], C_output_BGRA_[:,:,1], C_output_BGRA_[:,:,2], C_output_BGRA_[:,:,3])
        
        '''Estos histogramas corresponden a los analsis separados de los canales RGBA de la imagen original'''
        [E_hist_B_, E_hist_G_, E_hist_R_, E_hist_A_] = hist(E_output_B_, E_output_G_, E_output_R_, E_output_A_)
        

        #ASIGNAMOS LOS HISTOGRAMAS A CADA UNA DE LAS LINEAS DE LAS GRAFICAS
        C_line_B_.set_ydata(C_hist_B_)#asignamos los datos de hist_B_ a line_B_
        C_line_G_.set_ydata(C_hist_G_)#asignamos los datos de hist_G_ a line_G_
        C_line_R_.set_ydata(C_hist_R_)#asignamos los datos de hist_R_ a line_R_
        C_line_A_.set_ydata(C_hist_A_)#asignamos los datos de hist_A a line_A_
        C_line_B.set_ydata(hist_B)#asignamos los datos de hist_B a line_B
        C_line_G.set_ydata(hist_G)#asignamos los datos de hist_G a line_G
        C_line_R.set_ydata(hist_R)#asignamos los datos de hist_R a line_R
        C_line_A.set_ydata(hist_A)#asignamos los datos de hist_A a line_A
        
        
        #ASIGNAMOS LOS HISTOGRAMAS A CADA UNA DE LAS LINEAS DE LAS GRAFICAS
        E_line_B_.set_ydata(E_hist_B_)#asignamos los datos de hist_B_ a line_B_
        E_line_G_.set_ydata(E_hist_G_)#asignamos los datos de hist_G_ a line_G_
        E_line_R_.set_ydata(E_hist_R_)#asignamos los datos de hist_R_ a line_R_
        E_line_A_.set_ydata(E_hist_A_)#asignamos los datos de hist_A a line_A_
        E_line_B.set_ydata(hist_B)#asignamos los datos de hist_B a line_B
        E_line_G.set_ydata(hist_G)#asignamos los datos de hist_G a line_G
        E_line_R.set_ydata(hist_R)#asignamos los datos de hist_R a line_R
        E_line_A.set_ydata(hist_A)#asignamos los datos de hist_A a line_A
        
        
        #CALCULAMOS LOS MAXIMOS CON QUE LLEVAN C_ UNICAMENTE, PORQUE LOS DE E_ SON IGUALES
        [max_count_BGR, maxs_counts_BGR, maxs_intensity] = indentify_max(hist_B, hist_G, hist_R) 
        max_count_A = max(hist_A)
        
        #SETEAMOS LOS ASPECTOS VISUALES DE LOS ESPACIOS DE GRAFICACION DE LAS CURVAS DE ANALISIS CONJUNTO
        ax1.set_ylim(0, max_count_BGR)
        ax1.legend(loc='best')
        ax1.draw_artist(ax1.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax1.draw_artist(C_line_B_) #Redibuja line solo si es necesario
        ax1.draw_artist(C_line_G_) #Redibuja line solo si es necesario
        ax1.draw_artist(C_line_R_) #Redibuja line solo si es necesario

        
        ax2.set_ylim(0, max_count_A)
        ax2.legend(loc='best')
        ax2.draw_artist(ax2.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax1.draw_artist(C_line_A_) #Redibuja line solo si es necesario
        
        
        fig1.canvas.draw()#updatea el grafico con la nueva curva
        fig1.canvas.flush_events()#Hace un sleep para que se pueda crear la grafica
        
        
        #SETEAMOS LOS ASPECTOS VISUALES DE LOS ESPACIOS DE GRAFICACION DE LAS CURVAS DE ANALISIS POR SEPARADO
        ax3.set_ylim(0, max_count_BGR)
        ax3.legend(loc='best')
        ax3.draw_artist(ax1.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax3.draw_artist(E_line_B) #Redibuja line solo si es necesario

        ax4.set_ylim(0, max_count_A)
        ax4.legend(loc='best')
        ax4.draw_artist(ax2.patch)#selecciona al arreglo de ax, para darle rango al eje x
        ax4.draw_artist(E_line_A) #Redibuja line solo si es necesario
        
        fig2.canvas.draw()#updatea el grafico con la nueva curva
        fig2.canvas.flush_events()#Hace un sleep para que se pueda crear la grafica

    
    
    cv2.imshow('Gray scale selection', C_output_BGRA_[:,:,3])#Separacion de canal V)
    cv2.imshow('BGRA scale selection', C_output_BGRA_) #Mostramos la imagen
    
    if k == ord('s'):#grardamos los resultados
        
        [C_max_count_BGR_, C_maxs_counts_BGR_ , C_maxs_intensity_] = indentify_max(C_hist_B_, C_hist_G_, C_hist_R_) 
        C_max_count_A_ = max(C_hist_A_)
        #este arreglo x es para poder escribir los calores de intensidad dentro de archivo
        x = np.linspace(0.0, len(hist_A), len(hist_A))
        
        #calculamos los valores promedio, el valor de DS y el FWHM de cada uno
        [mean_B, sigma_B] = MeanAndSigma(x, hist_B)
        FWHM_B = FWHM(sigma_B) 
        [mean_G, sigma_G] = MeanAndSigma(x, hist_G)
        FWHM_G = FWHM(sigma_G) 
        [mean_R, sigma_R] = MeanAndSigma(x, hist_R)
        FWHM_R = FWHM(sigma_R) 
        [mean_A, sigma_A] = MeanAndSigma(x, hist_A)
        FWHM_A = FWHM(sigma_A)
        
        [C_mean_B_, C_sigma_B_] = MeanAndSigma(x, C_hist_B_)
        C_FWHM_B_ = FWHM(C_sigma_B_) 
        [C_mean_G_, C_sigma_G_] = MeanAndSigma(x, C_hist_G_)
        C_FWHM_G_ = FWHM(C_sigma_G_) 
        [C_mean_R_, C_sigma_R_] = MeanAndSigma(x, C_hist_R_)
        C_FWHM_R_ = FWHM(C_sigma_R_) 
        [C_mean_A_, C_sigma_A_] = MeanAndSigma(x, C_hist_A_)
        C_FWHM_A_ = FWHM(C_sigma_A_)
        #pdb.set_trace()
        
        print("Channel B:") 
        print("Complete max counts: {}".format(maxs_counts_BGR[0]))
        print("Section max counts: {}".format(C_maxs_counts_BGR_[0]))
        
        print("Channel G:") 
        print("Complete max counts: {}".format(maxs_counts_BGR[1]))
        print("Section max counts: {}".format(C_maxs_counts_BGR_[1]))
        
        print("Channel R:") 
        print("Complete max counts: {}".format(maxs_counts_BGR[2]))
        print("Section max counts: {}".format(C_maxs_counts_BGR_[2]))
        
        print("Channel A:") 
        print("Complete max counts: {}".format(max_count_A))
        print("Section max counts: {}".format(C_max_count_A_))
        
        data_file_name = "Complete_histograms_{}".format(file_name)
        data_file_name_ = "Section_histograms_{}".format(file_name)
        
        #creamos el folder general que guardara los resultados
        Fold_Creator('Results_BGRA_Analisis_'+ file_name + '_' + str(i))   
        Folder_Change('Results_BGRA_Analisis_'+ file_name + '_' + str(i))
        
        #escribimos  el archivo que contiene los histogramas de los canales completos 
        File_Writer(data_file_name,
                    [maxs_counts_BGR[0], maxs_counts_BGR[1], maxs_counts_BGR[2], max_count_A],
                    [maxs_intensity[0], maxs_intensity[1], maxs_intensity[2], hist_A.argmax()],
                    [hist_B, hist_G, hist_R, hist_A], 
                    [FWHM_B, FWHM_G, FWHM_R, FWHM_A])
        
        
        #escribimos el archivo que contiene los histogramas de las secciones
        File_Writer(data_file_name_, 
                    [C_maxs_counts_BGR_[0], C_maxs_counts_BGR_[1], C_maxs_counts_BGR_[2], C_max_count_A_],
                    [C_maxs_intensity_[0], C_maxs_intensity_[1], C_maxs_intensity_[2], C_hist_A_.argmax()],
                    [C_hist_B_, C_hist_G_, C_hist_R_, C_hist_A_], 
                    [C_FWHM_B_, C_FWHM_G_, C_FWHM_R_, C_FWHM_A_])
        
        #guardamos las graficas
        fig1.savefig('Complete_Histogram_BGR_' + file_name + '_' +str(i) + '.svg')
        fig2.savefig('Separate_Histogram_BGR_' + file_name + '_' +str(i) + '.svg')
        #fig2.savefig('Histogram_Grey_' + file_name + '_' +str(i) + '.svg')
        
        #guardamos las imagenes
        cv2.imwrite("Selection_" + data_file_name_ + '_B_channel' + str(i) +'_.jpg', C_output_BGRA_[:,:,0]) 
        cv2.imwrite("Selection_" + data_file_name_ + '_G_channel' + str(i) +'_.jpg', C_output_BGRA_[:,:,1]) 
        cv2.imwrite("Selection_" + data_file_name_ + '_R_Channel' + str(i) +'_.jpg', C_output_BGRA_[:,:,2]) 
        cv2.imwrite("Selection_" + data_file_name_ + '_A_Channel' + str(i) +'_.jpg', C_output_BGRA_[:,:,3]) 
        cv2.imwrite("Selection_" + data_file_name_ + '_' + str(i) +'_.jpg', C_output_BGRA_) 
        
        
        Folder_Change('..')
        i =+1





            
    
cv2.destroyAllWindows()#Destruimos todas las ventanas