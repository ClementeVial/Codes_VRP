# -*- coding: utf-8 -*-
"""Tesis copy.ipynb

# Tesis

### Descripción del modelo

**Instancia:**

* Se tiene un área de servicio donde un set de varios vehículos parten y terminan en un depósito. Los clientes realizan pedidos hasta un t_max.

* Los vehículos (con capacidad y autonomía ilimitada) tienen una velocidad determinística o estocástica durante cada tramo que recorren. Cada vez que recorre un tramo entre dos localizaciones se samplea una velocidad (con una distribución de probabilidad como lognormal si se está trabajando con el caso estocástico). Esto implica que la duración en los tiempos de viaje puede ser estocástica.

* Existen clientes tempranos (t_arrival = 0) que son conocidos al inicio del problema y clientes tardíos (t_arrival > 0) que van apareciendo dinámicamente durante el día, es decir, en instantes y ubicaciones aleatorias. Todos los clientes tienen un tiempo de servicio t_service estocástico. Además, cada cliente tiene una ventana de tiempo de 30 min desde que se confirmó su pedido para ser atendido sin penalización. Por último, cada cliente pertenece a una categoría que indica el nivel de reward que entrega al ser atendido.

* Nota: todos los clientes tienen un id único para cada realización.

**Estados:**

* Un estado tiene los siguientes elementos:

    * Plan de ruta de cada vehículo (que incluye la posición actual del vehículo en la posición 0 de la lista, y el depot en la posición final de la lista).

    * Un (posible) cliente aleatorio.
    
    * Tiempo t.

* En el estado inicial (t = 0) los vehículos se encuentran en el depot y el cliente aleatorio es uno de los clientes tempranos (t_arrival = 0).

* En el estado terminal los vehículos se encuentran en el depot, no quedan clientes por atender, y todos los clientes del día fueron vistos (confirmados o rechazados).
    
* Los puntos de decisión se dan en los momentos en que llega un nuevo cliente o bien luego de un intervalo de tiempo t_delta en el que no ha llegado ningún cliente.

**Acciones**

* Una acción en un punto de decisión incluye la confirmación o rechazo del cliente aleatorio y una decisión de movimiento; actualización del plan de ruta a cada vehículo.

* Las acciones posibles son:

    * Si llegó un cliente, insertarlo en alguna de las 6 posiciones menos costosas de alguno de los 4 vehículos más cercanos. (en el caso de los clientes iniciales se consideran los 4 vehículos con menor ocupación).

    * Si llegó un cliente rechazarlo y seguir con el plan de ruta de cada vehículo.

    * Si no llegó un cliente se puede seguir con el plan de ruta.

    * Algunas restricciones: los vehículos pueden esperar en el último cliente visitado si es que no tienen clientes pendientes por atender y no se ha superado t_max. Un cliente aleatorio sólo puede ser insertado en un vehículo. Todos los clientes tempranos (t_arrival = 0) deben ser atendidos, es decir, no existe la opción de rechazar a dichos clientes.

**Estado-Accion:**

* Un objeto State-Action tiene los siguientes elementos (dado que pertenece a una clase hija de State):

    * Plan de ruta de cada vehículo (que incluye la posición actual del vehículo en la posición 0 de la lista, y el depot en la posición final de la lista).

    * Un (posible) cliente aleatorio.
    
    * Tiempo t.

* Este objeto permite calcular los valores asociados a tomar una acción en cierto estado. Para esto se extrae de este elemento el vector de features.

**Transición y Rewards:**

* La transición entre dos estados está dada por la llegada de un cliente aleatorio o bien por el paso de t_delta minutos sin la llegada de un cliente. En este intervalo de tiempo se ejecuta el plan de ruta de cada vehículo, que considera para cada tramo entre localizaciones el sampleo de una velocidad que es aleatoria.

* En la transición a otro estado se percibe un reward que está compuesto por la ganancia asociada a aceptar al cliente menos la penalización por los clientes que fueron o están siendo atendidos fuera de su ventana de tiempo.

* La penalización por minuto de atraso está dada por una función monótona no decreciente (ej. raiz cuadrada).

**Algoritmos para la solución del MDP:**

1. Política miope 1 (cheapest insertion algorithm):

    * Para este algoritmo el take action considera lo siguiente: aceptar el cliente y asignarlo al vehículo más cercano en la posición menos costosa en su plan de ruta. En el ruteo de clientes iniciales se asigna el cliente al vehículo con la menor ocupación en la posición menos costosa.

2. Política inteligente (MC algorithm):

    * Este algoritmo considera un método Monte Carlo con un Value Function Approximation Lineal. Para promover la exploración, se utiliza una estrategia epsilon-greedy.

    * features: para el VFA lineal se utilizan 24 features polinomiales que incluyen información espacial y temporal del problema.

    * para actualizar los parámetros de la regresión de forma eficiente se utiliza RLS for stationary data.

    * Nota: El algoritmo del código está basado en el algoritmo que plantea Sutton en la sección 5.4, p. 104 (On policy first visit MC control for e-soft policies), utilizando RLS según los planteado por Powell en la sección 9.3.1 p. 350.

3. Política miope 2 (cluster algorithm):

    * Este algoritmo considera la clusterización del área de servicio en zonas fijas donde existe un grupo de vehículos asignado exclusivamente a cada zona.

    * Para este algoritmo el take action considera los siguiente: siempre aceptar al cliente aleatorio y, dependiendo de su zona, insertarlo con cheapest insertion (en términos de distancia recorrida) en alguna posición dentro de cualquiera de los planes de ruta de alguno de los vehículos asignados a su zona.

___

### Importación de librerías
"""

# librerías generales
import numpy as np
import copy
import random
import itertools
from math import log, sqrt, ceil
import heapq

# librerías para graficar y animar
import matplotlib.pyplot as plt
# import seaborn as sns

# para entregar parámetros por terminal
import argparse

"""### Funciones Auxiliares"""

def euclideanDistance(loc_list):

    '''
    Descripción:

        * Función auxiliar para medir la distancia euclideana de una secuencia de localizaciones del área de servicio. Por ejemplo, para una secuencia [o1, o2, o3], entrega
        la distancia euclideana entre o1 y o2 + la distancia euclideana entre o2 y o3. Si se entregan dos localizaciones, se mide la distancia sólo entre los dos objetos.

    Parámetros:

        * loc_list: secuencia de localizaciones (objetos con atributo posición)

    Return:

        * distance: distancia euclideana entre las localizaciones entregadas.
    '''

    # crear un array de coordenadas
    coord = np.array([loc.pos for loc in loc_list])

    # calcular la distancia total a lo largo de la secuencia de objetos
    delta = coord[1:] - coord[:-1]
    distances = np.hypot(delta[:, 0], delta[:, 1])
    total_distance = np.sum(distances)

    return total_distance

def sampleVel(instance):

    '''
    Descripción:

        * Función auxiliar para samplear la velocidad de un vehículo en un tramo (según el tipo de instancia puede ser determinista o estocástica).

    Parámetros:

        * instance: objeto de la clase Instance.

    Return:

        * vel: velocidad generada para el vehículo en el tramo.
    '''

    mean = log(instance.vel_mean ** 2 / (instance.vel_mean ** 2 + instance.vel_std ** 2) ** (1/2))
    std = (log(1 + (instance.vel_std ** 2 / instance.vel_mean ** 2))) ** (1/2)
    vel = round(random.lognormvariate(mean, std), 1)

    return vel

def sampleServiceTime(instance):

    '''
    Descripción:

        * Función auxiliar para samplear el tiempo de servicio de un cliente (según el tipo de instancia puede ser determinista o estocástica).

    Parámetros:

        * instance: objeto de la clase Instance.

    Return:

        * t_service: tiempo de servicio (en minutos) generado para el cliente.
    '''

    mean = log(instance.t_service_mean ** 2 / (instance.t_service_mean ** 2 + instance.t_service_std ** 2) ** (1/2))
    std = (log(1 + (instance.t_service_std ** 2 / instance.t_service_mean ** 2))) ** (1/2)
    t_service = round(random.lognormvariate(mean, std), 1)

    return t_service

def sampleArrivalTime(mean, std):

    '''
    Descripción:

        * Función auxiliar para samplear el tiempo de servicio de un cliente (según el tipo de instancia puede ser determinista o estocástica).

    Parámetros:

        * instance: objeto de la clase Instance.

    Return:

        * t_service: tiempo de servicio (en minutos) generado para el cliente.
    '''

    mean_ad = log(mean ** 2 / (mean ** 2 + std ** 2) ** (1/2))
    std_ad = (log(1 + (std ** 2 / mean ** 2))) ** (1/2)
    t_arrival = round(random.lognormvariate(mean_ad, std_ad), 1)

    return t_arrival

def mean_absolute_percentage_error(values_observed, values_predicted):

    '''
    Descripción:

        * Función auxiliar para calcular el error absoluto porcentual medio.

    Parámetros:

        * values_observed: secuencia con los valores observados (True).
        * values_predicted: secuencia con los valores predichos.

    Return:

        * mape: mean absolute percentage error.
    '''

    ape = [abs(value_observed - value_predicted) / max(1, abs(value_observed)) for value_observed, value_predicted in zip(values_observed, values_predicted)]
    mape = np.mean(ape)

    return mape

def mean_squared_error(values_observed, values_predicted):

    '''
    Descripción:

        * Función auxiliar para calcular el error cuadrático medio.

    Parámetros:

        * values_observed: secuencia con los valores observados (True).
        * values_predicted: secuencia con los valores predichos.

    Return:

        * mse: mean squared error.
    '''

    se = [(value_observed - value_predicted) ** 2 for value_observed, value_predicted in zip(values_observed, values_predicted)]
    mse = np.mean(se)

    return mse

"""### Clase Location"""

class Location:

    '''
    Descripción:

        * Clase para cualquier punto del área de servicio que no es depot o customer.

    Atributos:

        * id: 'Service Area'.
        * pos: coordenadas de posición.
        * t_start_serving: tiempo en que el vehículo llega a la posición.
        * t_leave: tiempo en que el vehículo deja la posición.
    '''

    def __init__(self):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * None.

        Return:

            * None.
        '''

        self.id = 'Service Area'
        self.pos = None
        self.t_start_serving = None
        self.t_leave = None

"""### Clase Depot"""

class Depot:

    '''
    Descripción:

        * Clase para puntos del área de servicio que son depot.

    Atributos:

        * id: 'Depot'.
        * pos: coordenadas de posición.
        * t_start_serving: tiempo en que el vehículo llega a la posición.
        * t_leave: tiempo en que el vehículo deja la posición.
    '''

    def __init__(self):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * None.

        Return:

            * None.
        '''

        self.id = 'Depot'
        self.pos = None
        self.t_start_serving = None
        self.t_leave = None

"""### Clase Customer"""

class Customer:

    '''
    Descripción:

        * Clase que genera objetos customer.

    Atributos:

        * id: id único para el objeto customer en la realización.
        * pos: coordenadas de posición.
        * t_arrival: tiempo en que aparece el cliente.
        * category: valor numérico que representa el nivel de importancia del cliente (debería ser igual o mayor a 1).
        * t_confirmed: si se confirma el cliente este atributo indica el instante en que se confirmó.
        * t_start_serving: si se confirma el cliente este atributo indica el instante en que se comenzó a atender.
        * t_leave: si se confirma el cliente este atributo indica el instante en que se dejó su posición.
        * status: puede ser 'not seen', 'confirmed' o 'rejected'.
        * penalty: indica la penalización acumulada que se ha generado por atender al cliente fuera de su ventana de tiempo.
    '''

    def __init__(self):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * None.

        Return:

            * None.
        '''

        self.id = None
        self.pos = None
        self.t_arrival = None
        self.category = None
        self.t_confirmed = None
        self.t_start_serving = None
        self.t_leave = None
        self.status = None
        self.penalty = None

"""### Clase Vehicle"""

class Vehicle:

    '''
    Descripción:

        * Clase que genera objetos vehicle.

    Atributos:

        * id: id único para el objeto vehicle.
        * route: lista que representa el plan de ruta del vehículo. El primer objeto de la lista indica la posición actual del vehículo, mientras que el objeto final de la lista es el depot.
    '''

    def __init__(self):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * None.

        Return:

            * None.
        '''

        self.id = None
        self.route = None

"""### Clase Instance"""

class Instance:

    '''
    Descripción:

        * Clase que genera la estructura básica del VRP a resolver y contiene los parámetros de la simulación.

    Atributos:

        * A_x: tamaño (en km) en la dirección x del área de servicio.
        * A_y: tamaño (en km) en la dirección y del área de servicio.
        * n_vehicles: número de vehículos.
        * n_cust_mean: número de clientes promedio.
        * dod: grado de dinamismo (degree of dinamism) del problema. Entendido como la proporción: Clientes tardíos / Clientes totales.
        * t_max: límite de tiempo (en min) para la llegada de pedidos.
        * t_service_mean: tiempo de servicio promedio (en min) para los clientes.
        * t_service_std: desviación estándar (en min) del tiempo de servicio para los clientes.
        * t_window: ventana de tiempo (en min) para la atención satisfactoria del cliente.
        * cust_categories: posibles categorías de cliente.
        * penalty_factor: factor de penalización por atención fuera de ventana de tiempo.
        * t_delta: tiempo máximo (en min) entre puntos de decisión. Si no llega un cliente en t_delta minutos se transita a un nuevo punto de decisión.
        * vel_mean: velocidad promedio de los vehículos (en km/min).
        * vel_std: desviación estándar de la velocidad de los vehículos (en km/min).
        * unif_distr: booleano que indica si la distribución de los clientes es uniforme (True) o clusterizada (False).

        * self.depot: objeto de la clase Depot que indica donde comienzan y terminan el día los vehículos.
        * vehicles: lista de objetos vehicles.
    '''

    def __init__(self, A_x, A_y, n_vehicles, n_cust_per_veh_mean, dod, t_max, t_service_mean, t_service_std, t_window, cust_categories, penalty_factor, t_delta, vel_mean, vel_std, unif_distr):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * A_x: tamaño en la dirección x del área de servicio. Se ingresa en km. 
            * A_y: tamaño en la dirección y del área de servicio. Se ingresa en km. 
            * n_vehicles: número de vehículos.
            * n_cust_per_veh_mean: número promedio de clientes por vehículo.
            * dod: número entre 0 y 1 que indica el grado de dinamismo.
            * t_max: límite de tiempo para la llegada de pedidos. Se ingresa en minutos.
            * t_service_mean: tiempo de servicio promedio para los clientes. Se ingresa en minutos.
            * t_service_std: desviación estándar del tiempo de servicio para los clientes. Se ingresa en minutos.
            * t_window: ventana de tiempo para la atención satisfactoria del cliente. Se ingresa en minutos.
            * cust_categories: lista que contiene las posibles categorías de cliente. Las categorías son numéricas.
            * penalty_factor: factor de penalización por atención fuera de ventana de tiempo. Un factor de 1/x indica que se alcanza una penalización de 1 cliente si se atiende con x minutos de atraso.
            * t_delta: tiempo máximo entre puntos de decisión. Se ingresa en minutos.
            * vel_mean: velocidad promedio de los vehículos. Se ingresa en km/h.
            * vel_std: desviación estándar de la velocidad de los vehículos. Se ingresa en km/h.
            * unif_distr: booleano que indica si la distribución de los clientes es uniforme (True) o clusterizada (False).

        Return:

            * None.
        '''

        # se definen los atributos de la clase
        self.A_x = A_x
        self.A_y = A_y
        self.n_vehicles = n_vehicles
        self.n_cust_mean = n_cust_per_veh_mean * n_vehicles
        self.dod = dod
        self.t_max = t_max
        self.t_service_mean = t_service_mean
        self.t_service_std = t_service_std
        self.t_window = t_window
        self.cust_categories = cust_categories
        self.penalty_factor = penalty_factor
        self.t_delta = t_delta
        self.vel_mean = vel_mean / 60
        self.vel_std = vel_std / 60
        self.unif_distr = unif_distr

        # se crea el depot
        self.depot = Depot()
        self.depot.pos = (self.A_x/2, self.A_y/2)

        # se crean los vehículos
        self.vehicles = []
        for i in range(self.n_vehicles):
            vehicle = Vehicle()
            vehicle.id = 'v' + str(i)
            self.vehicles.append(vehicle)

"""### Clase State"""

class State:

    '''
    Descripción:

        * Clase que genera objetos state. Estos objetos representan los estados que se generan en el sistema con la llegada de un cliente o con el paso de un tiempo determinado.

    Atributos:

        * vehicles: lista de los vehículos generados con Instance. Permite acceder al plan de ruta de los vehículos.
        * random_cust: objeto customer que corresponde al cliente que llega aleatoriamente entre un estado y otro.
        * t: instante de tiempo (en min) actual del episodio.
    '''

    def __init__(self, instance):

        '''
        Descripción:

            * Método constructor de la clase State.

        Parámetros:

            * instance: objeto de la clase Instance.

        Return:

            * None
        '''

        self.vehicles = instance.vehicles
        self.random_cust = None
        self.t = 0


    def initialState(self, realization, instance):

        '''
        Descripción:

            * Método que define el estado inicial del problema.

        Parámetros:

            * realization: lista de clientes de una realización aleatoria.
            * instance: objeto de la clase Instance.

        Return:

            * None
        '''

        # ningun cliente ha sido visto, por ende, su status es 'None', su penalty actual es 0 y no hay tiempos de inicio y fin de servicio aun
        for cust in realization:
            cust.t_confirmed = None
            cust.t_start_serving = None
            cust.t_leave = None
            cust.status = None
            cust.penalty = 0

        # el depot tiene tiempo de inicio y fin de servicio igual a 0 porque es la posición inicial
            instance.depot.t_start_serving = 0
            instance.depot.t_leave = 0

        # la ruta inicial de cada vehículo sólo contiene el depot como posición inicial y final
        for vehicle in self.vehicles:
            vehicle.route = [instance.depot, instance.depot]

        # el primer random customer es el primer cliente que tiene t_arrival = 0
        for cust in realization:
            if cust.t_arrival == 0:
                self.random_cust = cust
                break

        # tiempo inicial del problema
        self.t = 0


    def isTerminalState(self, instance):

        '''
        Descripción:

            * Método que permite identificar un estado terminal.

        Parámetros:

            * instance: objeto de la clase Instance.

        Return:

            * True o False dependiendo de si el estado es terminal o no.
        '''

        # si todos los vehículos tienen el depot como única localización en el plan de ruta entonces es el estado terminal
        for vehicle in self.vehicles:
            if vehicle.route != [instance.depot]:
                return False

        return True

"""### Clase StateAction"""

class StateAction(State):

    '''
    Descripción:

        * Clase que genera objetos state-action, que corresponde a un estado luego de haber aplicado una acción. Sobre estos objetos se calcula el value-to-go.

    Atributos:

        * vehicles: lista de los vehículos generados con Instance. Permite acceder al plan de ruta de los vehiculos.
        * random_cust: objeto customer que corresponde al cliente que llega aleatoriamente.
        * t: instante de tiempo (en min) actual del episodio.
    '''

    def __init__(self, state, action):

        '''
        Descripción:

            * Método constructor de la clase StateAction.

        Parámetros:

            * state: objeto de la clase State.
            * action: lista de vehículos que contienen sus respectivos planes de ruta.

        Return:

            * None
        '''

        self.state_deepcopy = copy.deepcopy(state)

        # la clase tiene los mismos atributos que el objeto state que toma como parámetro
        self.vehicles = [copy.deepcopy(vehicle) for vehicle in state.vehicles]
        self.random_cust = copy.deepcopy(state.random_cust)
        self.t = state.t

        # cambia la ruta de cada vehículo según la acción
        for vehicle, route in zip(self.vehicles, action):
            vehicle.route = route

        # se cambia el status de random_cust a confirmed o a rejected dependiendo si se insertó o no
        if self.random_cust is not None:
            self.random_cust.status = 'rejected'
            for route in action:
                if self.random_cust.id in [cust.id for cust in route]:
                    self.random_cust.status = 'confirmed'
                    self.random_cust.t_confirmed = self.t
                    break
            if self.random_cust.status == 'confirmed':
                for vehicle in self.vehicles:
                    for cust in vehicle.route:
                        if cust.id == self.random_cust.id:
                            cust.status = 'confirmed'
                            cust.t_confirmed = self.t


    def getFeatures(self, instance):

        '''
        Descripción:

            * Método para obtener características del objeto state.

        Parámetros:

            * instance: objeto de la clase Instance.

        Return:

            * features: lista que representa un vector de características del objeto state.
        '''

        # feature constante
        c = [1]

        # tiempo restante de pedidos
        t_left = [instance.t_max - self.t]

        # variable binaria para la inserción del cliente
        insertion_binary = [0]
        if self.random_cust is not None:
            for vehicle in self.vehicles:
                if self.random_cust.id in [cust.id for cust in vehicle.route]:
                    insertion_binary = [1]
                    break

        # costo (en distancia) de la inserción
        route_len_pre, route_len_post = 0, 0
        for vehicle_pre, vehicle_post in zip(self.state_deepcopy.vehicles, self.vehicles):
            route_len_pre += euclideanDistance(vehicle_pre.route)
            route_len_post += euclideanDistance(vehicle_post.route)
        insertion_cost = [route_len_post - route_len_pre]

        # número de clientes por atender
        n_cust_pend = 0
        for vehicle in self.vehicles:
            n_cust_pend += sum(isinstance(cust, Customer) for cust in vehicle.route)
        n_cust_pend = [n_cust_pend]

        # desviación del largo (en número de localizaciones) de los planes de ruta
        route_nlen = [len(vehicle.route) for vehicle in self.vehicles]
        std_route_nlen = [np.std(route_nlen)]

        # promedio de la raiz cuadrada del largo (en número de localizaciones) de los planes de ruta
        sqrt_route_nlen = [np.sqrt(len(vehicle.route)) for vehicle in self.vehicles]
        mean_sqrt_route_nlen = [np.mean(sqrt_route_nlen)]

        # desviación de la raiz del largo (en número de localizaciones) de los planes de ruta
        std_sqrt_route_nlen = [np.std(sqrt_route_nlen)]

        # razón entre tiempo restante y el número de clientes por atender
        t_per_n_cust_pend = [t_left[0] / max(1, n_cust_pend[0])]

        # promedio del largo (en distancia) del plan de ruta
        route_len = [sum(euclideanDistance([vehicle.route[i], vehicle.route[i+1]]) for i in range(len(vehicle.route)-1)) for vehicle in self.vehicles if len(vehicle.route) > 1]
        mean_route_len = [np.mean(route_len)]

        # desviación del largo (en distancia) de los planes de ruta
        std_route_len = [np.std(route_len)]

        # producto entre el tiempo restante y el largo promedio de las rutas
        t_mean_route_len = [t_left[0] * mean_route_len[0]]

        # promedio de la distancia entre los vehículos
        vehicles_dist = []
        for vehicle_pair in itertools.combinations(self.vehicles, 2):
            dist = euclideanDistance([vehicle_pair[0].route[0], vehicle_pair[1].route[0]])
            vehicles_dist.append(dist)
        mean_vehicles_dist = [np.mean(vehicles_dist)]

        # promedio de la desviación de la distancia entre todos las localizaciones de las rutas
        std_vehicles_loc_dist = []
        for vehicle in self.vehicles:
            loc_dist = [euclideanDistance([a, b]) for a, b in itertools.combinations(vehicle.route, 2)]
            std_vehicles_loc_dist.append(np.std(loc_dist))
        mean_std_loc_dist = [np.mean(std_vehicles_loc_dist)]

        # promedio al cuadrado de la desviación de la distancia entre todos las localizaciones de las rutas
        squared_mean_std_loc_dist = [mean_std_loc_dist[0] ** 2]

        # promedio de la distancia máxima entre dos puntos en las rutas
        vehicles_max_dist = []
        for vehicle in self.vehicles:
            max_dist = 0
            for i, j in itertools.combinations(range(len(vehicle.route)), 2):
                dist = euclideanDistance([vehicle.route[i], vehicle.route[j]])
                if dist > max_dist:
                    max_dist = dist
        vehicles_max_dist.append(max_dist)
        mean_max_dist = [np.mean(vehicles_max_dist)]

        # promedio al cuadrado de la distancia máxima entre dos puntos en las rutas
        squared_mean_max_dist = [mean_max_dist[0] ** 2]

        # std del factor de ocupación
        vehicles_of = [np.sum([isinstance(loc, Customer) for loc in vehicle.route]) / max(1, n_cust_pend[0]) for vehicle in self.vehicles]
        std_vehicles_of = [np.std(vehicles_of)]

        # promedio del penalty en el sistema
        vehicles_penalties = []
        vehicles_delays = []
        for vehicle in self.vehicles:
            penalty = 0
            delay = 0
            for cust in vehicle.route:
                if cust.t_start_serving is None:
                    if isinstance(cust, Customer) and self.t >= cust.t_confirmed + instance.t_window:
                        extra_time = self.t - (cust.t_confirmed + instance.t_window)
                        penalty += round(sqrt(extra_time * instance.penalty_factor), 2)
                        delay += extra_time
                elif cust.t_start_serving is not None:
                    if isinstance(cust, Customer) and cust.t_start_serving >= cust.t_confirmed + instance.t_window:
                        extra_time = cust.t_start_serving - (cust.t_confirmed + instance.t_window)
                        penalty += round(sqrt(extra_time * instance.penalty_factor), 2)
                        delay += extra_time
            vehicles_penalties.append(penalty)
            vehicles_delays.append(delay)
        mean_penalty = [np.mean(vehicles_penalties)]

        # std del penalty en el sistema
        std_penalty = [np.std(vehicles_penalties)]

        # promedio de atraso en el sistema
        mean_delay = [np.mean(vehicles_delays)]

        # desviación del atraso en el sistema
        std_delay = [np.std(vehicles_delays)]

        # tiempo restante por promedio del atraso
        t_mean_delay = [t_left[0] * mean_delay[0]]

        # eficiencia promedio de los planes de ruta
        route_efficiency = [length / nlength for length, nlength in zip(route_len, route_nlen)]
        mean_route_efficency = [np.mean(route_efficiency)]

        # se crea el vector de features
        features = np.array([c +
                             t_left +
                             insertion_binary +
                             insertion_cost +
                             n_cust_pend +
                             std_route_nlen +
                             mean_sqrt_route_nlen +
                             std_sqrt_route_nlen +
                             t_per_n_cust_pend +
                             mean_route_len +
                             std_route_len +
                             t_mean_route_len +
                             mean_vehicles_dist +
                             mean_std_loc_dist +
                             squared_mean_std_loc_dist +
                             mean_max_dist +
                             squared_mean_max_dist +
                             std_vehicles_of +
                             mean_penalty +
                             std_penalty +
                             mean_delay +
                             std_delay +
                             t_mean_delay +
                             mean_route_efficency]).T

        return features

"""### Clase Process"""

class Process:

    '''
    Descripción:

        * Esta clase contiene la estructura del modelo; genera objetos que contienen el MDP.

    Atributos:

        * None.
    '''

    def __init__(self):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * None.

        Return:

            * None.
        '''

        pass


    def computeActions(self, state):

        '''
        Descripción:

            * Método para determinar las acciones posibles en un estado.

        Parámetros:

            * state: objeto de la clase State.

        Return:

            * actions: lista que contiene todas las acciones posibles para state.
        '''

        if state.random_cust is not None:
            if state.random_cust.t_arrival == 0:
                # se definen los vehículos con menor cantidad de clientes
                n_cust_list = []
                for vehicle in state.vehicles:
                    n_cust = len(vehicle.route)
                    n_cust_list.append((vehicle, n_cust))
                shorter = heapq.nsmallest(4, n_cust_list, key=lambda x: x[1])
                possible_vehicles = [vehicle for vehicle, _ in shorter]
            else:
                # se definen los vehículos más cercano al cliente random
                distances = []
                for vehicle in state.vehicles:
                    dist = euclideanDistance([vehicle.route[0], state.random_cust])
                    distances.append((vehicle, dist))
                closest = heapq.nsmallest(4, distances, key=lambda x: x[1])
                possible_vehicles = [vehicle for vehicle, _ in closest]

        # se obtienen las acciones para cada vehículo
        actions_per_vehicle = []
        for vehicle in state.vehicles:
            actions_one_vehicle = []

            # 1. insertar cliente en alguna ruta de vehículos posibles
            if state.random_cust is not None and vehicle in possible_vehicles:
                # se calcula el largo de las diferentes rutas que se generan por las distintas inserciones
                routes_len = []
                for i in range(1, len(vehicle.route)):
                    route_copy = copy.copy(vehicle.route)
                    route_copy.insert(i, state.random_cust)
                    length = euclideanDistance(route_copy)
                    routes_len.append((route_copy, length))
                # sólo se consideran las mejores inserciones como posibles acciones
                best_insertions = heapq.nsmallest(6, routes_len, key=lambda x: x[1])
                best_insertions_routes = [route for route, _ in best_insertions]
                for route in best_insertions_routes:
                    actions_one_vehicle.append(route)

            # 2. seguir con la ruta. Condiciones necesarias: vehicle.route no es del tipo ['service area', 'Depot'] porque no se puede esperar en "service area"
            if not (isinstance(vehicle.route[0], Location) and isinstance(vehicle.route[1], Depot)):
                actions_one_vehicle.append(vehicle.route)

            # se guardan las acciones del vehículo
            actions_per_vehicle.append(actions_one_vehicle)

        # se crean las acciones, donde por cada acción individual de un vehículo, los demás vehículos no toman acción de movimiento
        actions = []
        for i, actions_one_vehicle in enumerate(actions_per_vehicle):
            for route in actions_one_vehicle:
                action = [vehicle.route for vehicle in state.vehicles]
                if state.random_cust is None or state.random_cust in route or (state.random_cust not in route and state.random_cust.t_arrival != 0):
                    action[i] = route
                    actions.append(action)

        # se elimina la repetición de la accion donde todos los vehículos se mantienen con la misma ruta
        dic = {}
        for action in actions:
            key = str(action)
            dic[key] = action
        actions = list(dic.values())

        return actions


    def transition(self, state, action, realization, instance):

        '''
        Descripción:

            * Método que determina la transición aleatoria entre un estado y otro.

        Parámetros:

            * state: objeto de la clase State.
            * action: lista de vehículos que contienen sus respectivos planes de ruta.
            * realization: lista de clientes de una realización aleatoria.
            * instance: objeto de la clase Instance.

        Return:

            * state: estado actualizado luego de la transición. Corresponde a un objeto de la clase State.
            * reward: reward total (ganancias - penalizaciones) percibido en la transición.
        '''

        # actualización del random customer
        random_cust_before_transition = state.random_cust
        state.random_cust = None
        for cust in realization:
            if state.t <= cust.t_arrival <= state.t + instance.t_delta and cust.status is None:
                state.random_cust = cust
                break

        # actualización del tiempo del estado: cuando llega un cliente o pasados t_delta minutos si no ha llegado ningun cliente
        t_before_transition = state.t
        if state.random_cust is None:
            state.t += instance.t_delta
            # si luego de t_delta sin llegada de clientes se supera el t_max, significa que se deben atender los clientes pendientes y volver al depot
            if state.t > instance.t_max:
                state.t = float('inf')
        else:
            state.t = state.random_cust.t_arrival

        # actualización del plan de ruta de cada vehículo
        for vehicle, route in zip(state.vehicles, action):

            # cuando se supera el t_max se atienden los clientes restantes y se vuelve al depot, porque no llegan clientes después de t_max
            if state.t > instance.t_max:
                range_sup = len(route)
            # cuando aun no se supera t_max se llega sólo hasta el último cliente y se espera en esa posición, no se llega hasta el depot
            else:
                range_sup = len(route)-1

            # se itera sobre cada loc de la ruta
            for i in range(range_sup):
                # se samplea un velocidad para el tramo
                vel = sampleVel(instance)
                t_service = sampleServiceTime(instance)
                # si se está en el depot, el tiempo en que se deja la posición es t del estado previo a la transición
                if isinstance(route[i], Depot):
                    route[i].t_leave = t_before_transition
                # se define la variable time como el tiempo en que se deja la posición i
                time = route[i].t_leave
                # si time es menor a t y no es el último cliente de la ruta, se calcula la distancia entre la loc i e i+1
                if time < state.t and i != range_sup-1:
                    time += round(euclideanDistance([route[i], route[i+1]]) / vel, 1)
                    # si time considerando la duración del viaje entre i e i+1 es menor a t entonces se alcanza a llegar a i+1
                    if time < state.t:
                        # se define el tiempo en que se comienza a atender y se deja i+1
                        route[i+1].t_start_serving = time
                        # se define el tiempo en que se deja la posición
                        if isinstance(route[i+1], Customer):
                            route[i+1].t_leave = route[i+1].t_start_serving + t_service
                        else:
                            route[i+1].t_leave = route[i+1].t_start_serving
                    # si time considerando la duración del viaje entre i e i+1 no es menor a t entonces el vehículo queda en una localización entre i e i+1
                    else:
                        t_travel = state.t - route[i].t_leave
                        cos = (route[i+1].pos[0] - route[i].pos[0]) / euclideanDistance([route[i], route[i+1]])
                        sen = (route[i+1].pos[1] - route[i].pos[1]) / euclideanDistance([route[i], route[i+1]])
                        vel_x = vel * cos
                        vel_y = vel * sen
                        pos_x = round(route[i].pos[0] + vel_x * t_travel, 2)
                        pos_y = round(route[i].pos[1] + vel_y * t_travel, 2)
                        vehicle_loc = Location()
                        vehicle_loc.pos = (pos_x, pos_y)
                        vehicle_loc.t_start_serving = state.t
                        vehicle_loc.t_leave = vehicle_loc.t_start_serving
                        vehicle.route = [vehicle_loc] + route[i+1:]
                        break

                # si time es mayor o igual que t, el vehículo se encuentra en el punto i de la ruta
                elif time >= state.t:
                    vehicle.route = route[i:]
                    break

                # si se está en el último punto de la ruta, el vehículo se encuentra en el punto i de la ruta
                elif i == range_sup-1:
                    route[i].t_leave = state.t
                    vehicle.route = route[i:]
                    break

        # se calcula el reward de la transición
        # en primer lugar se considera la ganancia de haber aceptado al cliente o no
        reward = 0
        for route in action:
            if random_cust_before_transition in route:
                reward += random_cust_before_transition.category
                break
        # en segundo lugar se calcula la penalización por clientes atendidos fuera de su ventana de tiempo
        for route in action:
            for cust in route:
                extra_time = 0
                if cust.t_start_serving is None:
                    if isinstance(cust, Customer) and state.t > cust.t_confirmed + instance.t_window:
                        extra_time = state.t - (cust.t_confirmed + instance.t_window)
                elif cust.t_start_serving is not None:
                    if isinstance(cust, Customer) and cust.t_start_serving > cust.t_confirmed + instance.t_window:
                        extra_time = cust.t_start_serving - (cust.t_confirmed + instance.t_window)
                if extra_time > 0:
                    penalty = round(sqrt(extra_time * instance.penalty_factor), 2) - cust.penalty
                    reward -= penalty
                    cust.penalty += penalty

        return state, reward

"""### Clase ValueFunction"""

class ValueFunction:

    '''
    Descripción:

        * Clase que crea un objeto value function, que representa la aproximación lineal de la value function.

    Atributos:

        * weights: pesos de la regresión asociados a los features.
        * n_features: número de features que se extraen de un objeto stateaction para obtener la aproximación lineal.
        * B: matriz para la actualización de los pesos de la regresión. Este atributo sólo se crea cuando se utiliza el algoritmo RLS para actualizar la value function.

    '''


    def __init__(self, initial_weights=None):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * initial_weights: pesos iniciales de la regresión. Si no se ingresa un valor, se considera un vector de ceros.

        Return:

            * None.
        '''

        # se calcula la cantidad de features
        self.n_features = 24
        # se define el vector de pesos inicial
        if initial_weights is None:
            self.weights = np.zeros(self.n_features).reshape(1, -1).T
        else:
            self.weights = initial_weights.reshape(1, -1).T


    def initializeRecursiveLeastSquares(self, lambd):

        '''
        Descripción:

            * Método para la inicialización de la matriz de actualización B de Recursive Least Squares para el entrenamiento de la aproximación lineal de la value function.

        Parámetros:

            * lambd: parámetro de penalización (ridge) para la aproximación lineal de la value function.

        Return:

            * None.
        '''

        # inicialización de la matriz B para la actualización de los weights
        self.B = (1/lambd) * np.identity(self.n_features)


    def predict(self, features):

        '''
        Descripción:

            * Método que entrega el value-to-go aproximado para un stateaction.

        Parámetros:

            * features: features de un stateaction.

        Return:

            * value_pred: value predicho por la aproximación lineal de la value function.
        '''

        # se obtiene el value aproximado a partir de los parámetros actuales de la value function
        value_pred = np.dot(self.weights.T, features)[0][0]

        return value_pred


    def updateWeights(self, features, value_predicted, value_observed):

        '''
        Descripción:

            * Método que actualiza los pesos de la regresión.

        Parámetros:

            * features: features observados para un stateaction.
            * value_predicted: value-to-go predicho para el stateaction a partir de los features.
            * value_observed: value-to-go observado para el stateaction.

        Return:

            * None.
        '''

        # se actualiza gamma
        gamma = 1 + features.T @ self.B @ features
        # se calcula el error del value
        error = value_predicted - value_observed
        # se actualiza la matriz H
        H = (1 / gamma) * self.B
        # se actualizan los pesos
        self.weights = self.weights - (H @ features) * error
        # se actualiza la matriz B
        self.B = self.B - (1 / gamma) * (self.B @ features @ features.T @ self.B)

"""### Superclase Algorithm y subclases de algoritmos derivados"""

class Algorithm:
    
    '''
    Descripción:

        * Super clase que crea un objeto algorithm, que permite crear las realizaciones aleatorias.

    Atributos:

        * instance: objeto de la clase Instance que contiene las características del problema.
        * process: objeto de la clase Process que contiene el MDP.
    ''' 

    def __init__(self, instance, process):

        '''
        Descripción:

            * Método constructor de la clase.

        Parámetros:

            * instance: objeto de la clase Instance que contiene las características del problema.
            * process: objeto de la clase Process que contiene el MDP.

        Return:

            * None.
        '''

        self.instance = instance
        self.process = process

    
    def simulateUniformRealizations(self, N, simulation_seed):

        '''
        Descripción:

            * Método para crear simular realizaciones aleatorias.

        Parámetros:

            * N: n° de realizaciones a simular.
            * simulation_seed: seed para replicar las realizaciones generadas.

        Return:

            * simulated_realizations: realizaciones aleatorias.
        '''

        random.seed(simulation_seed)
        np.random.seed(simulation_seed)
        realizations = []
        for _ in range(N):
            n_cust = np.random.binomial(self.instance.n_cust_mean*2, 0.5)
            realization = []
            # se crean los early request customers (t_arrival = 0)
            for j in range(ceil((1-self.instance.dod)*n_cust)):
                customer = Customer()
                customer.id = 'C' + str(j)
                customer.pos = (round(random.uniform(0, self.instance.A_x), 2), round(random.uniform(0, self.instance.A_y), 2))
                customer.t_arrival = 0
                customer.category = random.choice(self.instance.cust_categories)
                realization.append(customer)
            # se crean los late request customers (t_arrival > 0)
            for j in range(ceil((1-self.instance.dod)*n_cust), n_cust):
                customer = Customer()
                customer.id = 'C' + str(j)
                customer.pos = (round(random.uniform(0, self.instance.A_x), 2), round(random.uniform(0, self.instance.A_y), 2))
                customer.t_arrival = round(random.uniform(1, self.instance.t_max), 1)
                customer.category = random.choice(self.instance.cust_categories)
                realization.append(customer)
            # se ordena la realización por tiempo de llegada de los clientes
            realization = sorted(realization, key=lambda cust : cust.t_arrival)
            realizations.append(realization)
        simulated_realizations = realizations

        return simulated_realizations
    

    def simulateClusterRealizations(self, N, simulation_seed):

        '''
        Descripción:

            * Método para crear simular realizaciones aleatorias.

        Parámetros:

            * N: n° de realizaciones a simular.
            * simulation_seed: seed para replicar las realizaciones generadas.

        Return:

            * simulated_realizations: realizaciones aleatorias.
        '''

        random.seed(simulation_seed)
        np.random.seed(simulation_seed)
        realizations = []
        for _ in range(N):
            n_cust = np.random.binomial(self.instance.n_cust_mean*2, 0.5)
            realization = []
            # se crean los early request customers (t_arrival = 0)
            for j in range(ceil((1-self.instance.dod)*n_cust)):
                customer = Customer()
                customer.id = 'C' + str(j)
                rand = random.choice([1, 2, 3])
                if rand == 1:
                    customer.pos = (max(0, min(self.instance.A_x, round(random.gauss(0.5*self.instance.A_x, 0.5), 2))), max(0, min(self.instance.A_y, round(random.gauss(0.75*self.instance.A_y, 0.5), 2))))
                elif rand == 2:
                    customer.pos = (max(0, min(self.instance.A_x, round(random.gauss(0.25*self.instance.A_x, 0.5), 2))), max(0, min(self.instance.A_y, round(random.gauss(0.25*self.instance.A_y, 0.5), 2))))
                else:
                    customer.pos = (max(0, min(self.instance.A_x, round(random.gauss(0.75*self.instance.A_x, 0.5), 2))), max(0, min(self.instance.A_y, round(random.gauss(0.25*self.instance.A_y, 0.5), 2))))
                customer.t_arrival = 0
                customer.category = random.choice(self.instance.cust_categories)
                realization.append(customer)
            # se crean los late request customers (t_arrival > 0)
            for j in range(ceil((1-self.instance.dod)*n_cust), n_cust):
                customer = Customer()
                customer.id = 'C' + str(j)
                rand = random.choice([1, 2, 3])
                if rand == 1:
                    customer.pos = (max(0, min(self.instance.A_x, round(random.gauss(0.5*self.instance.A_x, 0.5), 2))), max(0, min(self.instance.A_y, round(random.gauss(0.75*self.instance.A_y, 0.5), 2))))
                elif rand == 2:
                    customer.pos = (max(0, min(self.instance.A_x, round(random.gauss(0.25*self.instance.A_x, 0.5), 2))), max(0, min(self.instance.A_y, round(random.gauss(0.25*self.instance.A_y, 0.5), 2))))
                else:
                    customer.pos = (max(0, min(self.instance.A_x, round(random.gauss(0.75*self.instance.A_x, 0.5), 2))), max(0, min(self.instance.A_y, round(random.gauss(0.25*self.instance.A_y, 0.5), 2))))
                if random.random() <= 0.5:
                    customer.t_arrival = sampleArrivalTime(self.instance.t_max*0.25, 30)
                else:
                    customer.t_arrival = sampleArrivalTime(self.instance.t_max*0.75, 30)
                customer.t_arrival = max(1, min(self.instance.t_max, customer.t_arrival))
                customer.category = random.choice(self.instance.cust_categories)
                realization.append(customer)
            # se ordena la realización por tiempo de llegada de los clientes
            realization = sorted(realization, key=lambda cust : cust.t_arrival)
            realizations.append(realization)
        simulated_realizations = realizations

        return simulated_realizations

    def simulateTrainRealizations(self, N, simulation_seed=None):

        '''
        Descripción:

            * Método para crear las realizaciones de train.

        Parámetros:

            * N: n° de realizaciones a simular.
            * simulation_seed: seed para replicar las realizaciones generadas.

        Return:

            * None.
        '''
        
        if self.instance.unif_distr:
            self.train_realizations = self.simulateUniformRealizations(N, simulation_seed)
        else:
            self.train_realizations = self.simulateClusterRealizations(N, simulation_seed)
        

    def simulateTestRealizations(self, N, simulation_seed=None):

        '''
        Descripción:

            * Método para crear las realizaciones de train.

        Parámetros:

            * N: n° de realizaciones a simular.
            * simulation_seed: seed para replicar las realizaciones generadas.

        Return:

            * None.
        '''

        if self.instance.unif_distr:
            self.test_realizations = self.simulateUniformRealizations(N, simulation_seed)
        else:
            self.test_realizations = self.simulateClusterRealizations(N, simulation_seed)

class CheapestInsertion(Algorithm):

    '''
    Descripción:

        * Subclase que crea un objeto del algortimo Cheapest Insertion. Esta clase contiene la forma una forma de resolver el VRP.

    Atributos:

        * instance: objeto de la clase Instance que contiene las características del problema.
        * process: objeto de la clase Process que contiene el MDP.
        * train_realizations: conjunto de realizaciones para entrenar el algoritmo.
        * test_realizations: conjunto de realizaciones para testear el algoritmo.
    '''

    def takeAction(self, state, actions):

        '''
        Descripción:

            * Método para que selecciona una acción en cierto estado a partir de un conjunto de acciones.

        Parámetros:

            * state: objeto de la clase State.
            * actions: lista que contiene todas las acciones posibles para state.

        Return:

            * min_len_action: entrega la acción que tiene el menor costo en distancia recorrida. Corresponde a una lista que contiene los planes de ruta de cada vehículo.
        '''

        if state.random_cust is not None:
            if state.random_cust.t_arrival == 0:
                # se define el vehículo con menor cantidad de clientes
                min_route_len = float('inf')
                for i, vehicle in enumerate(state.vehicles):
                    route_len = len(vehicle.route)
                    if route_len < min_route_len:
                        chosen_vehicle_index = i
                        min_route_len = route_len
            else:
                # se define cuál es el vehículo más cercano al cliente random
                min_dist = float('inf')
                for i, vehicle in enumerate(state.vehicles):
                    dist = euclideanDistance([vehicle.route[0], state.random_cust])
                    if dist < min_dist:
                        chosen_vehicle_index = i
                        min_dist = dist

        # se calcula el costo de la acción con la inserción en el vehículo elegido y se obtiene el mínimo
        min_len = float('inf')
        for action in actions:
            # se consideran sólo las acciones donde se inserta el cliente en el vehículo más cercano
            if (state.random_cust is not None and state.random_cust in action[chosen_vehicle_index]) or state.random_cust is None:
                action_len = 0
                for route in action:
                    # se cuantifica la distancia total de la ruta del vehículo más cercano
                    action_len += euclideanDistance(route)
                if action_len < min_len:
                    min_len = action_len
                    min_len_action = action

        # se cambia el status de random_cust (del estado justo antes de la transición) a confirmed o a rejected dependiendo si se insertó o no
        if state.random_cust is not None:
            state.random_cust.status = 'rejected'
            state.random_cust.t_confirmed = None
            for route in min_len_action:
                # si se insertó el cliente en la acción su status es 'confirmed'
                if state.random_cust in route:
                    state.random_cust.status = 'confirmed'
                    state.random_cust.t_confirmed = state.t
                    break

        return min_len_action


    def test(self):

        '''
        Descripción:

            * Método que aplica la política de decisión CI a una realizacion.

        Parámetros:

            * None.

        Return:

            * test_reward: rewards obtenidos en las realizaciones test aplicando la política CI.
            * penalties: valor total de penalización.
            * n_delays: número de clientes atendidos con retraso.
            * delays: atraso en minutos de todos los clientes en todas las realizaciones.
            * n_rejects: número de clientes rechazados.
        '''

        test_rewards = []
        penalties = []
        n_delays = []
        delays = []
        n_rejects = []

        for realization in copy.deepcopy(self.test_realizations):

            # se crea una instancia de state
            state = State(self.instance)
            # se crea variable para calcular el reward del episodio
            episode_reward = 0
            # se define el estado inicial
            state.initialState(realization, self.instance)

            while True:

                # si se está en el estado terminal termina el problema
                if state.isTerminalState(self.instance):
                    break
                # se buscan las acciones posibles en el estado
                actions = self.process.computeActions(state)
                # se toma la decisión devolviendo la accion
                action = self.takeAction(state, actions)
                # a partir de la acción y el estado se genera la transición al siguiente estado
                state, reward = self.process.transition(state, action, realization, self.instance)
                # se actualiza el reward total
                episode_reward += reward

            # se guarda el reward total
            test_rewards.append(episode_reward)
            # se guarda el penalty de la realización
            penalties.append(sum(cust.penalty for cust in realization))
            # se guarda la cantidad de pedidos atrasados
            n_delays.append(sum(cust.penalty > 0 for cust in realization))
            # se guardan los atrasos de todos los clientes
            delays += [cust.t_start_serving - cust.t_arrival - self.instance.t_window for cust in realization if cust.t_start_serving - cust.t_arrival > self.instance.t_window]
            # se guarda la cantidad de rechazos de la realización
            n_rejects.append(sum(cust.status == 'rejected' for cust in realization))

        return test_rewards, penalties, n_delays, delays, n_rejects

class OnPolicyMonteCarlo(Algorithm):

    '''
    Descripción:

        * Subclase que crea un objeto del algortimo AVI. Esta clase contiene la forma una forma de resolver el VRP.

    Atributos:

        * instance: objeto de la clase Instance que contiene las características del problema.
        * process: objeto de la clase Process que contiene el MDP.
        * train_realizations: conjunto de realizaciones para entrenar el algoritmo.
        * test_realizations: conjunto de realizaciones para testear el algoritmo.
    '''

    def takeAction(self, state, actions, value_function, train=False, epsilon=None):

        '''
        Descripción:

            * Método para que selecciona una acción en cierto estado a partir de un conjunto de acciones. Para esto, se calcula el value (con los parámetros actuales de la value function) de cada par estado-accion (representado por un objeto state_action) y se escoge una acción con una política epsilon-greedy (en el caso de entrenamiento) y greedy en el caso en que se aplique la política entrenada.

        Parámetros:

            * state: objeto de la clase State que representa el estado actual del problema.
            * actions: lista que contiene todas las acciones posibles para state.
            * value_function: objeto de la clase ValueFunction que contiene los pesos de la aproximación lineal.
            * train: booleano que indica si se está en entrenamiento o testeo, y así determinar si se utiliza o no epsilon greedy.
            * epsilon: hiperparámetro de epsilon greedy que indica la probabilidad de tomar una acción random.

        Return:

            * action: entrega la acción escogida (si está en fase de entrenamiento la escoge con epsilon-greedy). Corresponde a una lista que contiene los planes de ruta de cada vehículo.
        '''

        # se computan los state-action posibles y los values (predichos) asociados a cada uno
        best_value = float('-inf')
        for action in actions:
            # se crea un objeto state_action
            state_action = StateAction(state, action)
            # se extraen sus features
            features = state_action.getFeatures(self.instance)
            # se calcula el value
            value = value_function.predict(features)
            # se almacenan la acciones que da el mayor value
            if value > best_value:
                best_value = value
                best_action = action

        # si se está en fase de entrenamiento se considera una política epsilon-greedy
        if train:
            # random.seed(1387498)
            # con probabilidad epsilon se toma una acción aleatoria
            if random.random() < epsilon:
                action = random.choice(actions)
            # con probabilidad 1 - epsilon se toma una acción greedy
            else:
                action = best_action
        # si no se está en fase de entrenamiento se toma una acción greedy
        else:
            action = best_action

        # se cambia el status de random_cust (del estado justo antes de la transición) a confirmed o a rejected dependiendo si se insertó o no
        if state.random_cust is not None:
            state.random_cust.status = 'rejected'
            for route in action:
                if state.random_cust.id in [cust.id for cust in route]:
                    state.random_cust.status = 'confirmed'
                    state.random_cust.t_confirmed = state.t
                    break

        return action


    def train(self, value_function, epsilon):

        '''
        Descripción:

            * Método que aplica el algoritmo On policy Monte Carlo control (AVI) para encontrar una aproximación de la value function.

        Parámetros:

            * value_function: objeto de la clase ValueFunction para ser entrenado.
            * epsilon: hiperparámetro de epsilon greedy que indica la probabilidad de tomar una acción random.

        Return:

            * value_function: corresponde a un objeto de la clase ValueFunction que contiene un conjunto de pesos entrenados para la regresión que representa la aproximación lineal de la value function.
            * historic_weights: lista con el historial del vector de pesos de la regresión.
            * train_rewards: rewards para las realizaciones train.
            * penalties: valor total de penalización.
            * n_penalties: número de clientes atendidos con retraso.
            * n_rejects: número de clientes rechazados.
            * mape_list: lista de errores mape para cada episodio de entrenamiento.
            * mse_list: lista de errores mse para cada episodio de entrenamiento.
        '''

        mape_list = []
        mse_list = []
        historic_weights = []
        train_rewards = []
        penalties = []
        n_delays = []
        delays = []
        n_rejects = []

        for realization in copy.deepcopy(self.train_realizations):
            # se crea una instancia de state
            state = State(self.instance)
            # lista para guardar los rewards recibidos por luego de tomar una acción en un estado (rewards de un state_action)
            realization_rewards = []
            # listas para guardar los values observados y los values predichos de la realización
            realization_values_obs = []
            realization_values_pred = []
            # lista para guardar los features extraidos de los state_action
            realization_features = []
            # se define el estado inicial
            state.initialState(realization, self.instance)

            while True:
                # si se está en el estado terminal termina el problema
                if state.isTerminalState(self.instance):
                    break
                # se definen las acciones
                actions = self.process.computeActions(state)
                # se aplica take action sobre el conjunto de acciones
                action = self.takeAction(state, actions, value_function, train=True, epsilon=epsilon)
                # se crea un objeto state_action
                state_action = StateAction(state, action)
                # se extraen los features del state_action
                features = state_action.getFeatures(self.instance)
                # se guardan los features
                realization_features.insert(0, features)
                # se aplica la transición aleatoria creando un estado nuevo y un reward
                state, reward = self.process.transition(state, action, realization, self.instance)
                # se guarda el reward observado
                realization_rewards.append(reward)

            # a continuación se calcula y almacena el value observado para cada state_action visitado durante la realizacion
            realization_values_obs = list(itertools.accumulate(realization_rewards[::-1]))
            # se predice el value y se actualiza
            for features, value_obs in zip(realization_features, realization_values_obs):
                # se calcula el value predicho para los features observados en el state_action
                value_pred = value_function.predict(features)
                # guardar pesos históricos de la regresión
                historic_weights.append(value_function.weights.T.tolist()[0])
                # a partir del value observado se actualizan los pesos de la regresión
                value_function.updateWeights(features, value_pred, value_obs)
                # se guardan los valores predichos
                realization_values_pred.append(value_pred)

            # se guarda el reward del episodio
            train_rewards.append(np.sum(realization_rewards))
            # se guarda el penalty de la realización
            penalties.append(sum(cust.penalty for cust in realization))
            # se guarda la cantidad de pedidos atrasados
            n_delays.append(sum(cust.penalty > 0 for cust in realization))
            # se guardan los atrasos de todos los clientes
            delays += [cust.t_start_serving - cust.t_arrival - self.instance.t_window for cust in realization if cust.t_start_serving is not None and cust.t_start_serving - cust.t_arrival > self.instance.t_window]
            # se guarda la cantidad de rechazos de la realización
            n_rejects.append(sum(cust.status == 'rejected' for cust in realization))

            # se calcula el error de la predicción y se guarda en las listas de errores
            mape = mean_absolute_percentage_error(realization_values_obs, realization_values_pred)
            mse = mean_squared_error(realization_values_obs, realization_values_pred)
            mape_list.append(mape)
            mse_list.append(mse)

        return value_function, historic_weights, train_rewards, penalties, n_delays, delays, n_rejects, mape_list, mse_list


    def test(self, value_function):

        '''
        Descripción:

            * Método que aplica la política de decisión a una realization dada la value function previamente entrenada.

        Parámetros:

            * value_function: objeto de la clase ValueFunction que permite acceder a la política de decisión a través de la aproximación de los value de cada par estado-acción. Corresponde a la value function entrenada.

        Return:

            * test_reward: rewards obtenidos en las realizaciones test aplicando la política entrenada.
            * penalties: valor total de penalización.
            * n_penalties: número de clientes atendidos con retraso.
            * n_rejects: número de clientes rechazados.
            * test_mape: mape obtenido en test.
            * test_mse: mse obtenido en test.
        '''

        all_values_obs = []
        all_features_obs = []
        test_rewards = []
        penalties = []
        n_delays = []
        delays = []
        n_rejects = []

        for realization in copy.deepcopy(self.test_realizations):
            # se crea una instancia de state
            state = State(self.instance)
            # lista para guardar los rewards recibidos por luego de tomar una acción en un estado (rewards de un state_action)
            realization_rewards = []
            # lista para guardar los features extraidos de los state_action
            realization_features = []
            # se define el estado inicial
            state.initialState(realization, self.instance)

            while True:

                # si se está en el estado terminal termina el problema
                if state.isTerminalState(self.instance):
                    break
                # se definen las acciones
                actions = self.process.computeActions(state)
                # se aplica take action sobre el conjunto de acciones
                action = self.takeAction(state, actions, value_function)
                # se crea un objeto state_action
                state_action = StateAction(state, action)
                # se extraen las features del state_action
                features = state_action.getFeatures(self.instance)
                # se almacenan los features en la lista
                realization_features.append(features)

                # se aplica la transición aleatoria creando un estado nuevo y un reward
                state, reward = self.process.transition(state, action, realization, self.instance)
                # se guarda el reward observado en la primera posición de lista
                realization_rewards.append(reward)

            # a continuación se calcula y almacena el value observado para cada state_action visitado durante la realizacion
            realization_values_obs = list(itertools.accumulate(realization_rewards[::-1]))[::-1]
            # se agregan los values observados en la realización a la lista de values observados
            all_values_obs += realization_values_obs
            # se agregan los features observados en la realización a la lista de features observados
            all_features_obs += realization_features
            # se guarda el reward del episodio
            test_rewards.append(np.sum(realization_rewards))
            # se guarda el penalty de la realización
            penalties.append(sum(cust.penalty for cust in realization))
            # se guarda la cantidad de pedidos atrasados
            n_delays.append(sum(cust.penalty > 0 for cust in realization))
            # se guardan los atrasos de todos los clientes
            delays += [cust.t_start_serving - cust.t_arrival - self.instance.t_window for cust in realization if cust.t_start_serving is not None and cust.t_start_serving - cust.t_arrival > self.instance.t_window]
            # se guarda la cantidad de rechazos de la realización
            n_rejects.append(sum(cust.status == 'rejected' for cust in realization))

        # a partir de los features observados se predicen los values
        values_pred = [value_function.predict(features) for features in all_features_obs]

        # se calculan los errores de entrenamiento
        test_mape = round(mean_absolute_percentage_error(all_values_obs, values_pred), 2)
        test_mse = round(mean_squared_error(all_values_obs, values_pred), 2)

        return test_rewards, penalties, n_delays, delays, n_rejects, test_mape, test_mse

"""
___
___
___
"""

# Se crean las 162 instancias para los experimentos, un Proceso y objetos de los Algoritmos de resolución (CI y OPMC)
# Las primeras 54 son las deterministas, luego las ts estocásticas y luego las ts tv estocásticas
# Dentro de cada batch de 54, las primeras 27 son unif y las otras 27 clusterizadas.
# Dentro de cada batch de 27, las primeras 9 son 0.8, después 0.9, después 0.99

instances = [
Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=0,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),







Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=0,
        unif_distr=False),







Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=True),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.80,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.90,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=6,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=8,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=20,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=30,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False),

Instance(A_x=6,
        A_y=6,
        n_vehicles=10,
        n_cust_per_veh_mean=40,
        dod=0.99,
        t_max=420,
        t_service_mean=7,
        t_service_std=3,
        t_window=30,
        cust_categories=[1],
        penalty_factor=1/15,
        t_delta=5,
        vel_mean=30,
        vel_std=5,
        unif_distr=False)
]

# genero un objeto argparse para definir la instancia que se utilizará
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--num_exp', required=True, help='Experiment number', type=int)
args = vars(ap.parse_args())

# se crea un objeto Process que contiene el MDP
process = Process()
# se crea un objeto de CI
cheapest_insertion = CheapestInsertion(instances[args['num_exp']], process)
# se crea un objeto de OPMC
monte_carlo = OnPolicyMonteCarlo(instances[args['num_exp']], process)

# se crea un archivo de texto para guardar los resultados del experimento

if instances[args['num_exp']].t_service_std == 0 and instances[args['num_exp']].vel_std == 0:
    if instances[args['num_exp']].unif_distr:
        results = open('results det unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.txt', 'w')
    else:
        results = open('results det clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.txt', 'w')
elif instances[args['num_exp']].t_service_std != 0 and instances[args['num_exp']].vel_std == 0:
    if instances[args['num_exp']].unif_distr:
        results = open('results stoc unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.txt', 'w')
    else:
        results = open('results stoc clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.txt', 'w')
elif instances[args['num_exp']].t_service_std != 0 and instances[args['num_exp']].vel_std != 0:
    if instances[args['num_exp']].unif_distr:
        results = open('results full stoc unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.txt', 'w')
    else:
        results = open('results full stoc clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.txt', 'w')

"""
### Cheapest Insertion (Test)

"""

# se crean las realizaciones de testeo para el algoritmo
cheapest_insertion.simulateTestRealizations(N=1000, simulation_seed=5)
# testeando la política básica
ci_test_rewards, ci_test_penalties, ci_test_ndelays, ci_test_delays, ci_test_nrejects = cheapest_insertion.test()

# se guardan resultados de CI test en el archivo de texto
results.write('RESULTADOS CHEAPEST INSERTION TEST\n')
results.write('Nr de clientes por realizacion: ' + str([len(realization) for realization in cheapest_insertion.test_realizations]) + '\n')
results.write('Rewards por realizacion: ' + str(ci_test_rewards) + '\n')
results.write('Penalties por realizacion: ' + str(ci_test_penalties) + '\n')
results.write('Nr de pedidos con atraso por realizacion: ' + str(ci_test_ndelays) + '\n')
results.write('Atrasos (en minutos): ' + str(ci_test_delays) + '\n')
results.write('Nr de pedidos rechazados por realizacion: ' + str(ci_test_nrejects) + '\n')
results.write('---\n')
results.write('Nr promedio de clientes: ' + str(round(np.mean([len(realization) for realization in cheapest_insertion.test_realizations]), 2)) + '\n')
results.write('Reward promedio: ' + str(round(np.mean(ci_test_rewards), 2)) + '\n')
results.write('Penalty promedio: ' + str(round(np.mean(ci_test_penalties), 2)) + '\n')
results.write('Nr promedio de pedidos con atraso: ' + str(round(np.mean(ci_test_ndelays), 2)) + '\n')
results.write('Atraso promedio (en minutos): ' + str(round(np.mean(ci_test_delays), 2)) + '\n')
results.write('Nr promedio de pedidos rechazados: ' + str(round(np.mean(ci_test_nrejects), 2)) + '\n')
results.write('\n')

"""___

### On Policy Monte Carlo Control (Train y Test)
"""

# se crean las realizaciones de entrenamiento para el algoritmo
monte_carlo.simulateTrainRealizations(N=10000, simulation_seed=9)
# se crea una instancia de value function
mc_initial_value_function = ValueFunction()
# se inicializa RLS para la aproximación de la value function
mc_initial_value_function.initializeRecursiveLeastSquares(lambd=1/1000)
# entrenando el modelo con el método que utiliza RLS
mc_trained_value_function, mc_train_historic_weights, mc_train_rewards, mc_train_penalties, mc_train_ndelays, mc_train_delays, mc_train_nrejects, mc_mape_list, mc_mse_list = monte_carlo.train(mc_initial_value_function, epsilon=0.05)

# se guarda el "intervalo al 95%" de cada peso de la regresión
bounds_list = []
mc_train_historic_weights_tail = mc_train_historic_weights[-250000:]
for i in range(24):
    weights_sorted = sorted(np.array(mc_train_historic_weights_tail).T[i].tolist())
    perc_2_5 = int(len(weights_sorted) * 0.025)
    perc_97_5 = int(len(weights_sorted) * 0.975)
    weights_range = weights_sorted[perc_2_5:perc_97_5]
    bounds = weights_range[0], weights_range[-1]
    bounds_list.append(bounds)

# se guardan los resultados en el archivo de texto
results.write('RESULTADOS MONTE CARLO TRAIN\n')
results.write('Pesos regresion: ' + str((mc_trained_value_function.weights.T).tolist()[0]) + '\n')
results.write('Bounds de los pesos de la regresion: ' + str(bounds_list) + '\n')
results.write('---\n')
results.write('Nr de clientes por realizacion: ' + str([len(realization) for realization in monte_carlo.train_realizations]) + '\n')
results.write('Rewards por realizacion: ' + str(mc_train_rewards) + '\n')
results.write('Penalties por realizacion: ' + str(mc_train_penalties) + '\n')
results.write('Nr de pedidos con atraso por realizacion: ' + str(mc_train_ndelays) + '\n')
results.write('Atrasos (en minutos): ' + str(mc_train_delays) + '\n')
results.write('Nr de pedidos rechazados por realizacion: ' + str(mc_train_nrejects) + '\n') 
results.write('---\n')
results.write('Nr promedio de clientes: ' + str(round(np.mean([len(realization) for realization in monte_carlo.train_realizations]), 2)) + '\n')
results.write('Reward promedio: ' + str(round(np.mean(mc_train_rewards), 2)) + '\n')
results.write('Penalty promedio: ' + str(round(np.mean(mc_train_penalties), 2)) + '\n')
results.write('Nr promedio de pedidos con atraso: ' + str(round(np.mean(mc_train_ndelays), 2)) + '\n')
results.write('Atraso promedio (en minutos): ' + str(round(np.mean(mc_train_delays), 2)) + '\n')
results.write('Nr promedio de pedidos rechazados: ' + str(round(np.mean(mc_train_nrejects), 2)) + '\n') 
results.write('\n')

# se grafica la evolución de los errores de entrenamiento
fig_error, axes = plt.subplots(2, 1, figsize = (10,5))
x = np.arange(0, len(mc_mape_list))
y_mape = mc_mape_list
axes[0].plot(x, y_mape)
axes[0].set_xlim(left=0)
axes[0].set_ylim(bottom=0)
axes[0].set_ylabel('Mean Absolute Percentage Error')
axes[0].set_xlabel('Run')
axes[0].set_title('Evolution of Train MAPE')
y_mse = mc_mse_list
axes[1].plot(x, y_mse)
axes[1].set_xlim(left=0)
axes[1].set_ylim(bottom=0)
axes[1].set_ylabel('Mean Squared Error')
axes[1].set_xlabel('Run')
axes[1].set_title('Evolution of Train MSE')
fig_error.tight_layout()
# se grafica la evolución de la norma de los pesos de la regresión
norm_list = []
for weigths in mc_train_historic_weights:
    norm_list.append(np.linalg.norm(weigths))
# se grafica la evolución de la norma
fig_weights, axes = plt.subplots(1, 1, figsize = (10,4))
x = range(len(norm_list))
axes.plot(x, norm_list)
axes.set_ylabel('Weight Vector Norm')
axes.set_xlabel('Update Run')
axes.set_title('Evolution of The Weight Vector Norm')

# se guardan las figuras de entrenamiento

if instances[args['num_exp']].t_service_std == 0 and instances[args['num_exp']].vel_std == 0:
    if instances[args['num_exp']].unif_distr:
        fig_error.savefig('error results det unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')
        fig_weights.savefig('weights results det unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')     
    else:
        fig_error.savefig('error results det clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')
        fig_weights.savefig('weights results det clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')       
elif instances[args['num_exp']].t_service_std != 0 and instances[args['num_exp']].vel_std == 0:
    if instances[args['num_exp']].unif_distr:
        fig_error.savefig('error results stoc unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')
        fig_weights.savefig('weights results stoc unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')   
    else:
        fig_error.savefig('error results stoc clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')
        fig_weights.savefig('weights results stoc clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')   
elif instances[args['num_exp']].t_service_std != 0 and instances[args['num_exp']].vel_std != 0:
    if instances[args['num_exp']].unif_distr:
        fig_error.savefig('error results full stoc unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')
        fig_weights.savefig('weights results full stoc unif ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')   
    else:
        fig_error.savefig('error results full stoc clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg')
        fig_weights.savefig('weights results full stoc clust ' + str(instances[args['num_exp']].dod) + ' ' + str(instances[args['num_exp']].n_vehicles) + '_' + str(int(instances[args['num_exp']].n_cust_mean / instances[args['num_exp']].n_vehicles)) + ' ' + '10000' + '.jpg') 

# se crean las realizaciones de test para el algoritmo
monte_carlo.simulateTestRealizations(N=1000, simulation_seed=5)
# testeando la política encontrada en train con el método RLS
mc_test_rewards, mc_test_penalties, mc_test_ndelays, mc_test_delays, mc_test_nrejects, mc_test_mape, mc_test_mse = monte_carlo.test(mc_trained_value_function)

# se guardan los resultados en el archivo de texto
results.write('RESULTADOS MONTE CARLO TEST\n')
results.write('Nr de clientes por realizacion: ' + str([len(realization) for realization in monte_carlo.test_realizations]) + '\n')
results.write('Rewards por realizacion: ' + str(mc_test_rewards) + '\n')
results.write('Penalties por realizacion: ' + str(mc_test_penalties) + '\n')
results.write('Nr de pedidos con atraso por realizacion: ' + str(mc_test_ndelays) + '\n')
results.write('Atrasos (en minutos): ' + str(mc_test_delays) + '\n')
results.write('Nr de pedidos rechazados por realizacion: ' + str(mc_test_nrejects) + '\n')
results.write('---\n')
results.write('Nr promedio de clientes: ' + str(round(np.mean([len(realization) for realization in monte_carlo.test_realizations]), 2)) + '\n')
results.write('Reward promedio: ' + str(round(np.mean(mc_test_rewards), 2)) + '\n')
results.write('Penalty promedio: ' + str(round(np.mean(mc_test_penalties), 2)) + '\n')
results.write('Nr promedio de pedidos con atraso: ' + str(round(np.mean(mc_test_ndelays), 2)) + '\n')
results.write('Atraso promedio (en minutos): ' + str(round(np.mean(mc_test_delays), 2)) + '\n')
results.write('Nr promedio de pedidos rechazados: ' + str(round(np.mean(mc_test_nrejects), 2)) + '\n')
results.write('Test mape: ' + str(mc_test_mape) + '\n')
results.write('Test mse: ' + str(mc_test_mse) + '\n')