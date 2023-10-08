import heartpy as hp
import matplotlib.pyplot as plt

data, timer = hp.load_exampledata(0)
plt.figure(figsize=(16,4))
plt.plot(data)
plt.show()

width, mt = hp.process(data, sample_rate = 100.0)
plt.figure(figsize=(16,4))
hp.plotter(width, mt)
for measure in mt.keys():
  print('%s: %f' %(measure, mt[measure]))

data, timer = hp.load_exampledata(1)
plt.figure(figsize=(16,4))
plt.plot(data)
plt.show()

width, mt = hp.process(data, sample_rate = 100.0)
plt.figure(figsize=(16,4))
hp.plotter(width, mt)
for measure in mt.keys():
  print('%s: %f' %(measure, mt[measure]))

data, timer = hp.load_exampledata(2)
print(timer[0])

sample_rate = hp.get_samplerate_datetime(timer, timeformat= '%Y-%m-%d %H:%M:%S.%f')
print("El ratio de ejemplo es: %F Hz" %sample_rate)

w, m = hp.process(data, sample_rate, report_time = True)

plt.figure(figsize=(12, 4))
hp.plotter(w, m)

plt.figure(figsize=(12, 4))
plt.xlim(20000, 30000)
hp.plotter(w, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))

#EJEMPLO 2

import heartpy as hp
import matplotlib.pyplot as plt

ratio_ejemplo = 250

datos = hp.get_data('e0103.csv')
plt.figure(figsize=(12, 4))
plt.plot(data)
plt.show()

wd, m = hp.process(datos, ratio_ejemplo)
plt.figure(figsize=(12, 4))
hp.plotter (wd,m)

for measure in m.keys():
    print('%s : %f' %(measure, m[measure]))

datos = hp.get_data('e0110.csv')

plt.figure(figsize=(12,4))
plt.plot(data)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(data[0:2500])
plt.show()

filtrado = hp.filter_signal(datos, cutoff = 0.05, sample_rate = ratio_ejemplo, filtertype = 'notch')

plt.figure(figsize=(12,4))
plt.plot(filtrado)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(datos[0:2500], label = 'señal original')
plt.plot(filtrado[0:2500], alpha = 0.5, label = 'señal filtrada')
plt.legend()
plt.show()

ancho, mt = hp.process(hp.scale_data(filtrado), ratio_ejemplo)

plt.figure(figsize=(12,4))
hp.plotter(ancho, mt)

for measure in mt.keys():
    print('%s: %f' %(measure, mt[measure]))

from scipy.signal import resample

datosnvos = resample(filtrado, len(filtrado)*2)
ancho, mt= hp.process(hp.scale_data(datosnvos), ratio_ejemplo * 2)

plt.figure(figsize=(12,4))
hp.plotter(ancho, mt)

for measure in mt.keys():
    print('%s : %f' %(measure, mt[measure]))

datos = hp.get_data('e0124.csv')

plt.figure(figsize=(12,4))
plt.plot(datos)
plt.show()

plt.figure(figsize=(12,4))
plt.plot(data[0:2500])
plt.show()

ancho, mt = hp.process(hp.scale_data(datos), ratio_ejemplo)

plt.figure(figsize=(12,4))
hp.plotter(ancho, mt)

for measure in mt.keys():
    print('%s : %f' %(measure, mt[measure]))

datosnvos = resample(datos, len(filtrado) * 2)

ancho, mt = hp.process(hp.scale_data(datosnvos), ratio_ejemplo * 2)

plt.figure(figsize=(12,4))
hp.plotter(ancho, mt)

for measure in mt.keys():
    print('%s : %f' %(measure, mt[measure]))

hp.plot_poincare(ancho, mt)

medida_pncre = ['sd1', 'sd2', 's', 'sd1/sd2']
print('\nMedidas de poincare no lineales:')
for measure in medida_pncre:
    print('%s: %f' %(measure, mt[measure]))

#EJEMPLO 3

import numpy as np
import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt

datos_smart = pd.read_csv('raw_ppg.csv')

datos_smart.keys()

plt.figure(figsize=(12,6))
plt.plot(datos_smart['ppg'].values)
plt.show()

señal = datos_smart['ppg'].values[14500:20500]
reloj = datos_smart['timer'].values[14500:20500]
plt.plot(señal)
plt.show()

timer[0:20]
help(hp.get_samplerate_datetime)
ratio_ejemplo = hp.get_samplerate_datetime(reloj, timeformat = '%H:%M:%S.%f')
print('El ratio de ejemplo es: %.3f Hz' %ratio_ejemplo)

from datetime import datetime

nuevoreloj = [datetime.strptime(x, '%H:%M:%S.%f') for x in reloj]
tmptranscurrido = []
for i in range(len(nuevoreloj) - 1):
    tmptranscurrido.append(1 / ((nuevoreloj[i+1] - nuevoreloj[i]).microseconds / 1000000 ))

plt.figure(figsize=(12,6))
plt.plot(tmptranscurrido)
plt.xlabel('Numero ejemplo')
plt.ylabel('Ejemplo actual de ratio en Hz')
plt.show()

print('Significa que el ratio de ejemplo es: %.3f' %np.mean(tmptranscurrido))
print('Ejemplo medio de ratio: %.3f' %np.median(tmptranscurrido))
print('Desviación estándar: %.3f' %np.std(tmptranscurrido))

plt.figure(figsize=(12,6))
plt.plot(señal[0:int(240 * ratio_ejemplo)])
plt.title('Señal original')
plt.show()

filtrado = hp.filter_signal(señal, [0.7, 3.5], sample_rate=ratio_ejemplo, order = 3, filtertype='bandpass')

plt.figure(figsize=(12,12))
plt.subplot(211)
plt.plot(señal[0:int(240 * ratio_ejemplo)])
plt.title('Señal original')
plt.subplot(212)
plt.plot(filtrado[0:int(240 * ratio_ejemplo)])
plt.title('Señal filtrada')
plt.show()

plt.figure(figsize=(12,6))
plt.plot(filtrado[0:int(ratio_ejemplo * 60)])
plt.title('Señal filtrada en segmento de 60 sgs')
plt.show()

from scipy.signal import resample

reejemplo = resample(filtrado, len(filtrado) * 10)
nvo_ratio_ejemplo = sample_rate * 10

for s in [[0, 10000], [10000, 20000], [20000, 30000], [30000, 40000], [40000, 50000]]:
    ancho, mt = hp.process(reejemplo[s[0]:s[1]], sample_rate = nvo_ratio_ejemplo,
                       high_precision=True, clean_rr=True)
    hp.plotter(ancho, mt, title = "Sección acercada", figsize = (12,6))
    hp.plot_poincare(ancho, mt)
    plt.show()
    for measure in mt.keys():
        print("%s: %f" %(measure, mt[measure]))

raw = datos_smart['ppg'].values

plt.plot(raw)
plt.show()

import sys
from scipy.signal import resample

tamaño = 100
std = []

for i in range(len(raw) // tamaño):
    inicio = i * tamaño
    fin = (i + 1) * tamaño
    prt = raw[inicio:fin]
    try:
        std.append(np.std(prt))
    except:
        print(i)

plt.plot(std)
plt.show()

plt.plot(raw)
plt.show()

plt.plot(raw[0:(len(raw) // tamaño) * tamaño] - resample(std, len(std) * tamaño))
plt.show()

(len(raw) // tamaño) * tamaño

max = np.max(raw)
min = np.min(raw)
rango_global = max - min

tamaño = 100
filtrado = []

for i in range(len(raw) // tamaño):
    inicio = i * tamaño
    fin = (i + 1) * tamaño
    prt = raw[inicio:fin]
    rng = np.max(prt) - np.min(prt)

    if ((rng >= (0.5 * rango_global))
        or
        (np.max(prt) >= 0.9 * max)
        or
        (np.min(prt) <= min + (0.1 * min))):

        for x in prt:
          filtrado.append(0)
    else:
        for x in prt:
            filtrado.append(x)

plt.figure(figsize=(12,6))
plt.plot(raw)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(filtrado)

#EJEMPLO 4

import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp

ratio_ejemplo = 32
datos = hp.get_data('ring_data.csv')

plt.figure(figsize=(12,6))
plt.plot(datos)
plt.show()

datos = np.nan_to_num(datos)

plt.figure(figsize=(12,4))
plt.plot(datos[0:(5*60) * ratio_ejemplo])
plt.ylim(15000, 17000)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(datos[(5 * 60) * ratio_ejemplo:(10 * 60) * ratio_ejemplo])
plt.show()

plt.figure(figsize=(12, 4))
plt.title("Grafica con zoom")
plt.plot(datos[(5 * 60) * ratio_ejemplo:(6 * 60) * ratio_ejemplo])
plt.show()

ppg_filtrado = hp.filter_signal(datos[(5 * 60) * ratio_ejemplo:(10 * 60) * ratio_ejemplo], cutoff = [0.8, 2.5], filtertype = "bandpass", sample_rate = ratio_ejemplo, order = 3, return_top= False )
plt.figure(figsize=(12, 4))
plt.plot(ppg_filtrado[0:((2*60)*32)])
plt.show()

ancho, mt = hp.process(ppg_filtrado, sample_rate = sample_rate, high_precision= True)

plt.figure(figsize=(12,4))
hp.plotter(ancho, mt)

for key in m.keys():
    print('%s: %f' %(key, mt[key]))

#REVISAR!!!!!!!!!!!!!!!!!!!!!!!!!
plt.figure(figsize=(12, 6))
plt.xlim(0, (1 * 60) * sample_rate)
hp.plotter(ancho, mt, title="Primer minuto")

plt.figure(figsize=(12, 6))
plt.xlim((1 * 60) * sample_rate, (2 * 60) * ratio_ejemplo)
hp.plotter(ancho, mt, title="Segundo minuto")

plt.figure(figsize=(12, 6))
plt.xlim((2 * 60) * ratio_ejemplo, (3 * 60) * ratio_ejemplo)
hp.plotter(wd, m, title="Tercer minuto")

plt.figure(figsize=(12, 6))
plt.xlim((3 * 60) * ratio_ejemplo, (4 * 60) * ratio_ejemplo)
hp.plotter(wd, m, title="Cuarto minuto")

plt.figure(figsize=(12, 6))
plt.xlim((4 * 60) * ratio_ejemplo, (5 * 60) * ratio_ejemplo)
hp.plotter(wd, m, title="Quinto minuto")

ancho, mt = hp.process(ppg_filtrado, sample_rate = sample_rate, high_precision = True, clean_rr = True)

plt.figure(figsize=(12, 6))
hp.plotter(ancho, mt)

for key in m.keys():
    print("%s: %f" %(key, m[key]))

hp.plot_poincare(ancho, mt)

#EJEMPLO 5

import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp

ratio_ejemplo = 360

def carga_visualizada(ficheros, anotaciones):
    ecg = hp.get_data(ficheros)
    annotations = hp.get_data(anotaciones)

    plt.figure(figsize=(12, 7))
    plt.plot(ecg)
    plt.scatter(annotations, [ecg[int(x)] for x in annotations], color="green")
    plt.show()

    #ahora con zoom
    plt.figure(figsize=(12, 4))
    plt.plot(ecg)
    plt.scatter(annotations, [ecg[int(x)] for x in annotations], color="green")
    plt.xlim(20000, 26000)
    plt.show()

    return ecg, annotations

ecg, annotations = carga_visualizada("118e24.csv", "118e24_ann.csv")

def filtrado_visualizado(datos, ratio_ejemplo):
    filtrado = hp.remove_baseline_wander(datos, ratio_ejemplo)

    plt.figure(figsize=(12, 3))
    plt.title("Señal con desviación de la línea base eliminada")
    plt.plot(filtrado)
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.title("Zoom en grafica de señal con desviación de la línea base eliminada")
    plt.plot(hp.scale_data(datos[200:1200]))
    plt.plot(hp.scale_data(filtrado[200:1200]))
    plt.show()

    return filtrado

filtrado = filtrado_visualizado(ecg, sample_rate)

ancho, m = hp.process(hp.scale_data(filtrado), sample_rate)

plt.figure(figsize=(12, 4))
hp.plotter(ancho, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))

hp.plot_poincare(ancho, m)

ecg, anotaciones = carga_visualizada("118e12.csv", "118e12_ann.csv")

filtrado = filtrado_visualizado(ecg, ratio_ejemplo)

ancho, m = hp.process(hp.scale_data(filtrado), sample_rate)

plt.figure(figsize=(12,4))
hp.plotter(ancho, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))

hp.plot_poincare(ancho, m)

from scipy.signal import resample

señal_reemplazada = resample(filtrado, len(filtrado) * 4)

ancho, m = hp.process(hp.scale_data(señal_reemplazada), sample_rate * 4)

plt.figure(figsize=(28, 14))
hp.plotter(ancho, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))

hp.plot_poincare(ancho, m)

poincare = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare:
    print('%s: %f' %(measure, m[measure]))

ecg, anotaciones = carga_visualizada('118e12.csv', '118e12_ann.csv')

filtrado = hp.enhance_ecg_peaks(hp.scale_data(ecg), ratio_ejemplo, aggregation='median', iterations=5)

plt.figure(figsize=(12, 4))
plt.plot(filtrado)
plt.show()

plt.figure(figsize=(12,4))
plt.title("Señal original")
plt.plot(hp.scale_data(ecg[15000:17000]), label="Datos originales")
plt.title("Señal procesada")
plt.plot(hp.scale_data(filtrado[15000:17000]), alpha=0.5, label="Datos procesados")
plt.legend()
plt.show()

señal_reemplazada = resample(filtrado, len(filtrado) * 10)

ancho, m = hp.process(hp.scale_data(señal_reemplazada), ratio_ejemplo * 10)

plt.figure(figsize=(12, 4))
hp.plotter(ancho, m)

for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))

hp.plot_poincare(ancho, m)

poincare_m = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_m:
    print('%s: %f' %(measure, m[measure]))

ecg, anotaciones = carga_visualizada('118e00.csv', '118e00_ann.csv')

filtrado = hp.enhance_ecg_peaks(hp.scale_data(ecg), ratio_ejemplo, aggregation='median', iterations=4)

plt.figure(figsize=(12, 4))
plt.plot(filtrado)
plt.show()

plt.figure(figsize=(12, 4))
plt.title("Señal original")
plt.plot(hp.scale_data(ecg[15000:17000]), label="Datos originales")

plt.title("Señal procesada")
plt.plot(hp.scale_data(filtrado[15000:17000]), alpha=0.5, label="Datos procesados")
plt.legend()
plt.show()

señal_reemplazada = resample(filtrado, len(filtrado) * 10)

ancho, m = hp.process(hp.scale_data(señal_reemplazada), ratio_ejemplo * 10)

plt.figure(figsize=(12, 4))
hp.plotter(ancho, m)

for measure in m.keys():
    print("%s: %f" %(measure, m[measure]))

hp.plot_poincare(ancho, m)

poincare_m = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_m:
    print("%s: %f" %(measure, m[measure]))

filtrado = hp.filter_signal(ecg[0:14500], 0.05, ratio_ejemplo, filtertype='notch')

ancho, m = hp.process(hp.scale_data(filtrado), ratio_ejemplo)

plt.figure(figsize=(12, 4))
hp.plotter(ancho, m)

for measure in m.keys():
    print("%s: %f" %(measure, m[measure]))

hp.plot_poincare(ancho, m)

poincare_m = ['sd1', 'sd2', 's', 'sd1/sd2']
for measure in poincare_m:
    print("%s: %f" %(measure, m[measure]))
