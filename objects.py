class Slice:
    def __init__(self, number_, switch_start_, switch_end_, packet_, throughput_, delay_):
        self.number = number_               # номер слайса
        self.qos_throughput = throughput_   # требования к пропускной способности слайса
        self.qos_delay = delay_             # требования к задержке слайса
        self.estimate_delay = 0.0           # оценка задержки
        self.switch_start = switch_start_   # начальный коммутатор для потока
        self.switch_end = switch_end_       # конечный коммутатор для потока
        self.flows_list = list()            # список потоков(исправил)
        self.packet_size = packet_          # размер пакетов, передаваемых в слайсе
        self.ports_set = set()          # множество портов, через которые проходят потоки слайса
        # self.used_sw = set()                # множество коммутаторов из sls_sw_set, на которых уже нельзя изменить qos
        self.leaves = []                    # список всех вершин времен
        self.tree = 0                       # дерево связности времен
        self.route_time_constraints = []    # список временных неравенст для потока
        # self.sw2port = dict()


class Flow:
    def __init__(self, number_, rho_a_, b_a_):
        self.number = number_    # номер потока
        self.rho_a = rho_a_      # скорость поступления трафика (для кривой нагрузки)
        self.b_a = b_a_          # всплеск трафика (для кривой нагрузки)
        self.path = None        # список коммутатор, через которые проходит поток
        self.paths = []          # список путей, через которые может проходить поток


class Port:
    def __init__(self, _start, _end, speed_, remain_speed_):
        # self.number = number_           # номер порта
        self.link = (_start, _end)      # два коммутатора, которые связывает порт
        self.priority_list = list()     # список приоритетов на порту
        self.physical_speed = speed_    # физическая пропускная способность канала
        self.remaining_bandwidth = remain_speed_  # остаточная пропускная способность канала
        # self.slice_priorities = dict()  # соотношение номера слайса и его приоритета


class Switch:
    def __init__(self, number_):
        self.number = number_           # номер коммутатора
        self.slice2port = dict()        # соотвествие порта и слайса на sw
        self.port_list = list()         # список выходных портов на коммутаторе  


class Priority:
    def __init__(self, number_, throughput_, qos_delay_, queue):
        self.priority = number_                     # значение приоритета
        self.throughput = throughput_               # пропускная способность приоритета
        self.queue_list = [queue]                   # список очередей, входящих в приоритет
        self.mean_delay = qos_delay_                # среднее требование по задержка приоритета
        self.delay = 0.0                            # суммарная задержка приоритета
        self.priority_lambda = queue.slice_lambda   # lambda суммарная по всем очередям
        self.sigma_priority = 0.0                   # сумма нагрузок вышестоящих приоритетов
        self.slice_queue = dict()                   # соотношение номер слайса и очереди

    def recalculation(self):
        self.mean_delay = 0.0
        self.priority_lambda = 0.0
        required_throughput = 0.0
        for queue in self.queue_list:
            self.mean_delay += queue.slice.qos_delay
            required_throughput += queue.slice.qos_throughput
            self.priority_lambda += queue.slice_lambda
        self.mean_delay /= len(self.queue_list)
        for queue in self.queue_list:
            queue.weight = queue.slice.qos_throughput / required_throughput
            queue.rho_s = queue.weight * self.throughput


class Queue:
    def __init__(self, slice_, priority_):
        self.number = slice_.number # номер очереди == номеру слайса
        self.priority = priority_   # приоритет очереди на порту
        self.weight = 1.0           # доля пропускной способности приоритета (omega)
        self.slice = slice_         # указать на слайс, который передает в этой очереди
        self.rho_s = 0.0            # скорость для кривой обслуживания
        self.b_s = 0.0              # задержка для кривой обслуживания
        # self.flow_numbers = 0       # количество маршрутов слайса, проходящих через этот коммутатор
        self.slice_lambda = 0.0     # lambda суммарная по всем потокам в слайсе
        # self.add_flow_number(sw)
        self.find_slice_input_flow()

    # def add_flow_number(self, sw):
    #     for flow in self.slice.flows_list:
    #         for elem in flow.path:
    #             if elem == sw:
    #                 self.flow_numbers += 1

    def find_slice_input_flow(self):
        for flow in self.slice.flows_list:
            self.slice_lambda += flow.rho_a

