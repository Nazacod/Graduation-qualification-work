import json 
import heapq
import typing
import itertools

import objects
from mytime import MyTime, Tree

# формируем начальный временной промежуток
def start_serv(time_t, elem):
    equal_start = dict()
    for i in range(len(time_t)):
        if time_t[i] != elem and time_t[i][0] == elem[0]:
            equal_start[i] = time_t[i]
    return equal_start


# проверяем подпоследовательности времен задачи ЛП
def equal_time(time_t, pos1, pos2):
    sub_pos1 = time_t.index(time_t[pos1][1:])
    sub_pos2 = time_t.index(time_t[pos2][1:])
    if sub_pos1 <= pos1 or sub_pos1 <= pos2:
        return False
    if sub_pos2 <= pos1 or sub_pos2 <= pos2:
        return False
    return True


# проверяем подпоследовательности времен задачи ЛП
def different_time(time_t, pos1, pos2):
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    sub_pos1 = time_t.index(time_t[pos1][1:])
    sub_pos2 = time_t.index(time_t[pos2][1:])
    if (sub_pos1 > pos1) and (sub_pos1 < pos2 < sub_pos2):
        return True
    return False


# проверяем корректно ли составлены временные неравенства задачи ЛП
def correct_time(time_t):
    swap = list()
    for i in range(len(time_t)):
        if time_t[len(time_t) - 1] != MyTime('0'):
            return False
        if time_t[i][1:] == '':
            continue
        if time_t.index(time_t[i][1:]) <= i:
            return False
        position = start_serv(time_t, time_t[i])
        for key in position.keys():
            equal = equal_time(time_t, i, key)
            if not equal and not different_time(time_t, i, key):
                return False
            if equal and i < key and time_t[i][1] > time_t[key][1]:
                swap.append(tuple((i, key)))
    for it in swap:
        time_t[it[0]], time_t[it[1]] = time_t[it[1]], time_t[it[0]]
    return True

# выбираем знак неравенства
def expression_sign(task, i):
    if task[i][0] == task[i + 1][0]:
        return ' = '
    return ' <= '

def my_sort(sorted_list, according_list):
    new_list = []
    index_list = []
    for elem in sorted_list:
        index_list.append(according_list.index(elem))
    index_list.sort()
    for index in index_list:
        new_list.append(according_list[index])
    return new_list

# записываем какие потоки проходят через данный коммутатор
def choose_routes(sw, flows_list):
    routes_in_sw = list()
    for flow in flows_list:
        if sw in flow.path:
            routes_in_sw.append(flow.number)
    return routes_in_sw

# записываем кумулетативную функцию одного сервера
def one_server(routes_in_sw, time_t, sw, elem):
    help_str = ""
    flag = False
    for route in routes_in_sw:
        if flag:
            help_str += (' ' + elem + ' ')
        help_str += ('F' + str(route) + 's' + str(sw) + 't' + time_t.elem)
        flag = True
    return help_str


class Graph:
    def __init__(self, edges: list[tuple[int, int, float]] = []):
        """
            edges - множество ребер графа со своим весом
        """
        self.adj_list = {}
        self.edge2port = {}
        self.number2switch = {}
        self.lengthLP = 0
        
        for start, end, throughput in edges:
            if start in self.adj_list:
                self.adj_list[start].append(end)
            else:
                self.adj_list[start] = [end]
            
            if end not in self.adj_list:
                self.adj_list[end] = []

            if start not in self.number2switch:
                self.number2switch[start] = objects.Switch(start)
            
            if end not in self.number2switch:
                self.number2switch[end] = objects.Switch(end)

            port = objects.Port(start, end, throughput, throughput)
            
            self.number2switch[start].port_list.append(port)
            self.edge2port[(start, end)] = port
    
    def add_edge(self, start: int, end: int, throughput: float, remain_speed):   
        if (start, end) in self.edge2port:
            return
        
        if start in self.adj_list:
            self.adj_list[start].append(end)
        else:
            self.adj_list[start] = [end]
                    
        if end not in self.adj_list:
                self.adj_list[end] = []

        if start not in self.number2switch:
                self.number2switch[start] = objects.Switch(start)
            
        if end not in self.number2switch:
            self.number2switch[end] = objects.Switch(end)

        port = objects.Port(start, end, throughput, remain_speed)
        
        self.number2switch[start].port_list.append(port)
        self.edge2port[(start, end)] = port
            
    
    def get_adjacent_vertices(self, vertex: int):
        return self.adj_list[vertex] if vertex in self.adj_list else []
    
    # создаем множество времен для потока, который вычисляем
    def form_flow_time(self, flow, sls):
        first_tree = Tree(MyTime('0'))
        end = flow.path[len(flow.path) - 1] #последний коммутатор для пути
        sls.tree = Tree(MyTime(end), first_tree)
        sls.leaves = [sls.tree]
        time_t = [MyTime(end)]
        vertex = [MyTime(end)]
        while vertex != list():
            elem = vertex.pop(0)
            for lk in sls.ports_set:
            # for lk in self.sls_ports_set:
                # if lk[0] not in sls.sls_sw_set:
                #     continue
                if elem[0] == str(lk[1]):
                    new_elem = MyTime(lk[0]) + elem
                    time_t.append(new_elem)
                    vertex.append(new_elem)

                    new_tree = Tree(new_elem, sls.tree)
                    sls.leaves.remove(sls.tree)
                    sls.leaves.append(new_tree)
                    sls.tree = new_tree
        time_t.append(MyTime('0'))
        # print(time_t)
        # print(sls.tree)
        return time_t
    
    # создаем множества времен для каждого свитча потока
    def form_switches_time(self, time_t, flows_list):
        routes_dict = dict()
        # number = 1
        for flow in flows_list:
            one_route = dict()
            input_variables = list()
            for sw in self.number2switch.keys():
                if sw not in flow.path:
                    continue
                one_switch = list()
                for t in time_t:
                    if t[0] != str(sw):
                        continue
                    one_switch.append(t)
                    if t not in input_variables:
                        input_variables.append(t)
                    if t[1:] in time_t:
                        one_switch.append(t[1:])
                        if t[1:] not in input_variables:
                            input_variables.append(t[1:])
                    elif t[1:] == '':
                        one_switch.append(MyTime('0'))
                        if MyTime('0') not in input_variables:
                            input_variables.append(MyTime('0'))
                one_route[str(sw)] = one_switch
            one_route['0'] = input_variables
            routes_dict[flow.number] = one_route
            # number += 1
        # print(routes_dict)
        return routes_dict
    
    # формируем список возможных задач для вычисляеого потока
    def time_constraints(self, time_t, sls):
        sls.route_time_constraints = list()
        constraints = list()
        pos = 0
        for i in range(len(time_t)):
            constraints.append('0')
        self.build_time_constraints(constraints, pos, sls)
        # print(sls.route_time_constraints)
        return sls.route_time_constraints

    # рекурсивно формируем список возможных задач для вычисляеого потока
    def build_time_constraints(self, constraints, pos, sls):
        if sls.leaves == list():
            if correct_time(constraints):
                if constraints not in sls.route_time_constraints:
                    sls.route_time_constraints.append(constraints)
        else:
            elem = sls.leaves.pop(0)
            if elem.next is not None:
                sls.leaves.append(elem.next)
            constraints[pos] = elem.value
            self.build_time_constraints(constraints, pos + 1, sls)

    # создаем все неравенства без учета положения задержки для одной задачи
    def create_lp(self, task, routes_dict, flows_list, sls):
        help_str = '//time\n'
        help_str += self.write_time(task)
        for route in flows_list:
            help_str += ("\n//flow " + str(route.number) + '\n')
            help_str += self.generate_constraints(task, routes_dict, route)
            help_str += ("\n//arrival " + str(route.number) + "\n")
            help_str += self.generate_arrival(task, route)
        help_str += "\n//servers\n"
        help_str += self.generate_servers(task, flows_list, sls)
        return help_str

    # записываем временные неравенства
    def write_time(self, task):
        help_str = ""
        for i in range(len(task) - 1):
            help_str += ('t' + task[i].elem + expression_sign(task, i) + 't' + task[i + 1].elem + ';\n')
            self.lengthLP += 1
        return help_str

    # генерируем и записываем неравенства маршрута
    def generate_constraints(self, task, routes_dict, flow):
        help_str = ""
        path = flow.number
        for sw in routes_dict[path]:
            time_t = my_sort(routes_dict[path][sw], task) if sw != '0' else task
            help_str += ('//sw ' + str(sw) + '\n')
            pos = 0
            for i in range(len(time_t) - 1):
                # non_decreasing functions
                help_str += ('F' + str(path) + 's' + sw + 't' + time_t[i].elem)
                help_str += (expression_sign(task, i))
                help_str += ('F' + str(path) + 's' + sw + 't' + time_t[i + 1].elem + ';\n')
                self.lengthLP += 1
                if sw == '0':
                    continue
                # start_backlog
                if sw == time_t[i][0]:
                    previos_pos = flow.path.index(int(sw))
                    previos_sw = '0' if previos_pos == 0 else str(flow.path[previos_pos - 1])
                    help_str += ('F' + str(path) + 's' + previos_sw + 't' + time_t[i].elem + ' = ')
                    help_str += ('F' + str(path) + 's' + sw + 't' + time_t[i].elem + ';\n')
                    self.lengthLP += 1
                # flow_contains
                else:
                    help_str += ('F' + str(path) + 's' + sw + 't' + time_t[i].elem + ' <= ')
                    help_str += ('F' + str(path) + 's0t' + time_t[i].elem + ';\n')
                    self.lengthLP += 1
                pos = i
            if sw != '0' and (pos + 1) < len(time_t):
                help_str += ('F' + str(path) + 's' + sw + 't' + time_t[pos + 1].elem + ' <= ')
                help_str += ('F' + str(path) + 's0t' + time_t[pos + 1].elem + ';\n')
                self.lengthLP += 1
        return help_str

    def generate_arrival(self, task, route):
        help_str = ""
        path = route.number
        for it in itertools.combinations(reversed(task), 2):
            if it[0] == it[1]:
                continue
            help_str += ('F' + str(path) + 's0t' + it[0].elem + ' - F' + str(path) + 's0t' + it[1].elem)
            help_str += (' <= ' + str(route.rho_a) + ' * t' + it[0].elem + ' - ' + str(route.rho_a) + ' * t' + it[1].elem + ' + ' + str(route.b_a) + ';\n')
            self.lengthLP += 1
        return help_str

    def generate_servers(self, task, flows_list, sls):
        help_str = ""
        for sw in self.number2switch.keys():
            eq_start = list()
            previos = 0
            routes_in_sw = choose_routes(sw, flows_list)
            # заполняем параметры кривой обслуживания
            rho_s = 0
            b_s = 0
            if sls.number in self.number2switch[sw].slice2port:
                port = self.edge2port[self.number2switch[sw].slice2port[sls.number]]
                for pr in port.priority_list:
                    flag = False
                    for queue in pr.queue_list:
                        if queue.number == sls.number:
                            rho_s = queue.rho_s
                            b_s = queue.b_s
                            flag = True
                            break
                    if flag:
                        break
            # на выходных вершинах скорость не ограничена
            if rho_s == 0:
                rho_s = 1000
            # if b_s == 0:
            #     b_s = 5
            # print('rho_s = ', rho_s, 'b_s = ', b_s)
            for time_t in task:
                if time_t == MyTime('0'):
                    continue
                if time_t[0] != str(sw):
                    continue
                t = MyTime('0') if time_t[1:] == '' else MyTime(time_t[1:])
                help_str += one_server(routes_in_sw, t, sw, '+')
                help_str += ' - '
                help_str += one_server(routes_in_sw, time_t, sw, '-')
                help_str += (' >= ' + str(rho_s) + ' * t' + t.elem + ' - ')
                help_str += (str(rho_s) + ' * t' + time_t.elem + ' - ' + str(b_s) + ';\n')
                self.lengthLP += 1
                if eq_start == list():
                    eq_start.append(time_t)
                    previos = task.index(time_t)
                elif task.index(time_t) - previos <= 1:
                    eq_start.append(time_t)
                    previos = task.index(time_t)
            help_str += self.servers_equal_start_time(eq_start, sw, routes_in_sw, rho_s, b_s)
        return help_str

    def servers_equal_start_time(self, eq_start, sw, routes_in_sw, rho_s, b_s):
        help_str = ""
        for it in itertools.combinations(eq_start, 2):
            help_str += one_server(routes_in_sw, MyTime(it[1][1:]), sw, '+')
            help_str += ' - '
            help_str += one_server(routes_in_sw, MyTime(it[0][1:]), sw, '-')
            help_str += (' >= ' + str(rho_s) + ' * t' + it[1][1:] + ' - ')
            help_str += (str(rho_s) + ' * t' + it[0][1:] + ' - ' + str(b_s) + ';\n')
            self.lengthLP += 1
        return help_str

    def write_delay_constraints(self, task, pos, flow, file):
        file.write('max: t0-u;\n\n')
        file.write('t' + task[pos - 1].elem + ' <= u;\n' + 'u <= t' + task[pos].elem + ';\n\n')
        file.write('F' + str(flow.number) + 's' + str(flow.path[len(flow.path) - 1]) + 't0 <= F' + str(
            flow.number) + 's0u;\n')
        self.lengthLP += 2
        file.write('\n//arrival delay \n')
        for time_t in task:
            if pos > task.index(time_t):
                file.write('F' + str(flow.number) + 's0u - F' + str(flow.number) + 's0t' + time_t.elem)
                elem1 = 'u'
                elem2 = 't' + time_t.elem
            else:
                file.write('F' + str(flow.number) + 's0t' + time_t.elem + ' - F' + str(flow.number) + 's0u')
                elem1 = 't' + time_t.elem
                elem2 = 'u'
            file.write(' <= ' + str(flow.rho_a) + ' * ' + elem1 + ' - ' + str(flow.rho_a) + ' * ' + elem2 + ' + ' + str(flow.b_a) + ';\n')
            self.lengthLP += 1
        file.write('\n')


def find_paths(graph: Graph, input_vertex: int, output_vertex: int):
    visited = set()
    path = []
    subgraph = Graph()
    _dfs(graph, input_vertex, output_vertex, visited, path, subgraph)
    return subgraph

def _dfs(graph: Graph, current_vertex: int, output_vertex: int, visited: set[int], path: list[int], subgraph: Graph):
    visited.add(current_vertex)
    path.append(current_vertex)
    
    if current_vertex == output_vertex:
        _add_path_to_subgraph(path, subgraph, graph)
    else:
        for neighbor in graph.get_adjacent_vertices(current_vertex):
            if neighbor not in visited:
                _dfs(graph, neighbor, output_vertex, visited, path, subgraph)
    
    path.pop()
    visited.remove(current_vertex)
    
def _add_path_to_subgraph(path: list[int], subgraph: Graph, graph: Graph):
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i+1]
        subgraph.add_edge(start, end, graph.edge2port[(start, end)].physical_speed, graph.edge2port[(start, end)].remaining_bandwidth)
        # print(subgraph.edge2port)


def beam_search(graph: Graph, start: int, goal: int, beam_width=1):
    queue = [(0, start, [start])]
    paths = []
    
    while queue:
        cost, current, path = heapq.heappop(queue)
        if current == goal:
            paths.append(path)
        
        for neighbor in graph.get_adjacent_vertices(current):
                new_path = path + [neighbor]
                # стоимость: длина + все веса на пути
                try:
                    # first
                    # new_cost = cost + 1
                    # second
                    # new_cost = cost + (1 / graph.edge2port[(new_path[-2], new_path[-1])].remaining_bandwidth)
                    # third
                    # new_cost = cost + (1 + 1 / graph.edge2port[(new_path[-2], new_path[-1])].remaining_bandwidth) 
                    # fourth
                    new_cost = max(cost, (1 / graph.edge2port[(new_path[-2], new_path[-1])].remaining_bandwidth))
                except ZeroDivisionError:
                    new_cost = float("inf")
                heapq.heappush(queue, (new_cost, neighbor, new_path))
        
        # print(queue)
        if len(queue) > beam_width:
            queue = heapq.nsmallest(beam_width, queue)
            # print('damn')
            # print(queue)

        if len(paths) >= beam_width:
            break

    return paths[:beam_width]


#вычислительно дорого 

# def oriented_graph_diameter(adj_dict):
#     # Get a list of unique node identifiers
#     node_ids = list(adj_dict.keys())

#     # Get the number of nodes in the graph
#     num_nodes = len(node_ids)

#     # Initialize the distance matrix with large values (representing infinity)
#     dist_matrix = [[float('inf') for _ in range(num_nodes)] for _ in range(num_nodes)]

#     # Set the distance to self as 0 for all nodes
#     for i in range(num_nodes):
#         dist_matrix[i][i] = 0

#     # Set the distance between directly connected nodes
#     for i, node in enumerate(node_ids):
#         neighbors = adj_dict[node]
#         for neighbor in neighbors:
#             j = node_ids.index(neighbor)
#             dist_matrix[i][j] = 1

#     # Apply the Floyd-Warshall algorithm to compute shortest paths between all pairs of nodes
#     for k in range(num_nodes):
#         for i in range(num_nodes):
#             for j in range(num_nodes):    
#                 dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])

#     # Find the diameter of the graph
#     diameter = 0
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if dist_matrix[i][j] != float('inf'):
#                 diameter = max(diameter, dist_matrix[i][j])

#     return diameter


# with open('./inputs/input_1.json') as json_file:
#     data = json.load(json_file)
#     links_with_weights = []
#     for elem in data["topology"]:
#         links_with_weights.append(tuple(elem.values()))
    
#     graph = Graph(links_with_weights)
#     print(graph.adj_list)
#     # print(graph.edge2port)
#     # print(graph.number2switch)
#     # print(len(graph.number2switch))
#     # print(len(graph.edge2port))
#     subgraph = find_paths(graph, 1, 6)
#     print(subgraph.adj_list)

#     paths = beam_search(subgraph, 1, 6, 1)
#     print(paths)

#     #print(oriented_graph_diameter(graph.adj_list))