import math
import random
import time
from copy import deepcopy

import slicedelay
import objects 
from topology import find_paths, beam_search

DELTA_DELAY = 0.8   #среднее отличие требовании задержки голосового трафика от видео

def create_queues(path, slice, graph):
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i+1] 
        port = graph.edge2port[(start, end)]        
        if port.remaining_bandwidth < slice.qos_throughput:
            # нет пропускной способности на таком маршруте
            return -1
        
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i+1] 
        # if (start, end) not in slice.sls_ports_set:
        
        port = graph.edge2port[(start, end)]        
        port.remaining_bandwidth -= slice.qos_throughput
        
        queue = objects.Queue(slice, 1)
        slice.ports_set.add((start, end))
        graph.number2switch[start].slice2port[slice.number] = (start, end)

        if len(port.priority_list) == 0:
            # создаем приоритет для порта
            priority = objects.Priority(1, slice.qos_throughput, slice.qos_delay, queue)
            priority.slice_queue[slice.number] = queue
            queue.rho_s = queue.weight * priority.throughput
            port.priority_list.append(priority)
            # topology.switches[sw].slice_priorities[sls] = priority.priority
        else:
            was_added = False
            for pr in port.priority_list:
                if math.fabs(pr.mean_delay - slice.qos_delay) < DELTA_DELAY:
                    was_added = True
                    # добавляем очередь в существующий приоритет
                    queue.priority = pr.priority
                    pr.queue_list.append(queue)
                    pr.slice_queue[slice.number] = queue
                    pr.throughput += slice.qos_throughput
                    pr.recalculation()
                    # topology.switches[sw].slice_priorities[sls] = pr.priority
            if not was_added:
                # создаем новый приоритет и туда добавляем очередь
                number = len(port.priority_list) + 1
                priority = objects.Priority(number, slice.qos_throughput, slice.qos_delay, queue)
                priority.slice_queue[slice.number] = queue
                queue.rho_s = queue.weight * priority.throughput
                port.priority_list.append(priority)
                # topology.switches[sw].slice_priorities[sls] = priority.priority
    return 0


def create_start_service_curve(graph):
    # на каждом порту вычисляем задержку приоритета
    for port in graph.edge2port.values():
        slicedelay.calculate_priority_delay(graph, port)

    # вычисляем задержку для каждой очереди
    for port in graph.edge2port.values():
        for pr in port.priority_list:
            slicedelay.calculate_queue_delay(pr)

def calc_delay(graph):
    # заглушка ;)
    return random.random()


def main_algorithm(graph, slices, slices_order, beam_width=8, filename='test'):
    # begin = time.time()
    while slices_order:
        current_slice_ind = slices_order.pop(0)
        if current_slice_ind > 44:
            pass
        current_slice = slices[current_slice_ind]
        switch_start, switch_end = current_slice.switch_start, current_slice.switch_end
        subgraph = find_paths(graph, switch_start, switch_end)

        paths = beam_search(subgraph, switch_start, switch_end, beam_width)
        # first_path = paths[0]

        current_slice.flows_list[0].paths = paths
        path2delay = {}
        for path in paths:
            graph_copy = deepcopy(graph)
            current_slice_copy = deepcopy(current_slice)
            current_slice_copy.flows_list[0].path = path
            res = create_queues(path, current_slice_copy, graph_copy)
            if res == 0:
                # формируем кривую обслуживания на каждом коммутаторе для начальных параметров
                create_start_service_curve(graph_copy)
                # считаем задержку на полученной конфигурации
                estimate_delay = slicedelay.calculate_slice_delay(current_slice_copy, graph_copy) #(graph_copy)
                path2delay[tuple(path)] = estimate_delay
        
        if len(path2delay) == 0:
            print('Нет путей вообще')
            print(f'Не можем установить {current_slice_ind} слайс')
            slices.pop(current_slice_ind)
            continue

        best_path, best_delay = sorted(path2delay.items(), key=lambda x: x[1])[0]

        if best_delay > current_slice.qos_delay:
            print('Не проход по задержке')
            print(f'Не можем установить {current_slice_ind} слайс')
            slices.pop(current_slice_ind)
            continue

        res = create_queues(best_path, current_slice, graph)
        create_start_service_curve(graph)
        
        current_slice.flows_list[0].path = best_path
        assert (res == 0)
        current_slice.estimate_delay = best_delay

    # with open(f"times/time_{filename}.txt", 'a') as file_out:
    #     print(time.time() - begin, file=file_out)
    # print('1')

