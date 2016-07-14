import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

graph = raw_input("Show errors graph (y/n): ")
response = str(graph).strip()
# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0] 
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",") #recoge los datos y los separa por comas
np.random.shuffle(data) #empareja
x_data = data[:,0:4].astype('f4') #4 primeras columnas que son las caracteristicas de la flor
y_data = one_hot(data[:,4].astype(int), 3) #ultima columna de iris.data, el resultado final

#print y_data

#print "\nSome samples..."
#for i in range(20):
#    print x_data[i], " -> ", y_data[i]
#print


DATOS_DE_IRIS = 4 #datos de cada flor (altura sepalo, anchura sepalo, altura petalo, anchura petalo)
SALIDA_NEURONAS = 3 #salida de las neuronas (posiciones del vector)

datos_entrada = tf.placeholder("float", [None, DATOS_DE_IRIS]) #reserva memoria
datos_salida = tf.placeholder("float", [None, SALIDA_NEURONAS]) #reserva memoria


NEURONAS_OCULTAS = 4 #nuevas neuronas

pesos_capa_oculta = tf.Variable(np.float32(np.random.rand(DATOS_DE_IRIS, NEURONAS_OCULTAS)) * 0.1) #prepara los pesos de la capa oculta
umbral_capa_oculta = tf.Variable(np.float32(np.random.rand(NEURONAS_OCULTAS)) * 0.1) #prepara los umbrales de la capa oculta

pesos_capa_visible = tf.Variable(np.float32(np.random.rand(NEURONAS_OCULTAS, SALIDA_NEURONAS)) * 0.1) # prepara los pesos para la capa visible
umbral_capa_visible = tf.Variable(np.float32(np.random.rand(SALIDA_NEURONAS)) * 0.1) #prepara los umbrales de la capa visible

salida_neuronas_capa_oculta = tf.sigmoid(tf.matmul(datos_entrada, pesos_capa_oculta) + umbral_capa_oculta) #funcion sigmoid, cuanto mas cerca este del 0 da 0 y cuanto mas cerca este del 1 da 1

salida_capa_visible = tf.nn.softmax((tf.matmul(salida_neuronas_capa_oculta, pesos_capa_visible) + umbral_capa_visible)) #prepara el valor final


cross_entropy = tf.reduce_sum(tf.square(datos_salida - salida_capa_visible)) #prepara el error: reduciendo la diferencia del valor que devuelven las neuronas y el real
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

errors = []
for step in xrange(1000):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj*batch_size : jj*batch_size+batch_size]
        batch_ys = y_data[jj*batch_size : jj*batch_size+batch_size]

        sess.run(train, feed_dict={datos_entrada: batch_xs, datos_salida: batch_ys})
        if step % 50 == 0:
            error = sess.run(cross_entropy, feed_dict={datos_entrada: batch_xs, datos_salida: batch_ys})
            errors.append(error)
            print "Iteration #:", step, "Error: ", error
            result = sess.run(salida_capa_visible, feed_dict={datos_entrada: batch_xs})
            for b, r in zip(batch_ys, result):
                print b, "-->", r
            print "----------------------------------------------------------------------------------"

if response == "y":
    plt.plot(errors)
    plt.show()

            
            
            
