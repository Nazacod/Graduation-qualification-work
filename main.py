import sys
import json
import topology
import objects
from algorithm import main_algorithm

def parse_config(filename):
    with open(filename) as json_file:
        # заполняем граф
        data = json.load(json_file)
        links_with_weights = []
        for elem in data["topology"]:
            links_with_weights.append(tuple(elem.values()))
    
        graph_ = topology.Graph(links_with_weights)

        # заполняем слайсы
        slices_ = {}
        for sls_data in data["slices"]:
            tmp_slice = objects.Slice(sls_data["sls_number"],
                                    sls_data["switch_start"],
                                    sls_data["switch_end"],
                                    sls_data["packet_size"],
                                    sls_data["qos"]["throughput"],
                                    sls_data["qos"]["delay"])
            
            tmp_flow = objects.Flow(sls_data["flow"]["flow_number"], sls_data["flow"]["rho_a"], sls_data["flow"]["b_a"])
            tmp_slice.flows_list.append(tmp_flow)
            slices_[sls_data["sls_number"]] = tmp_slice

        return graph_, slices_

def write_result(file_name, slices, graph):
    result = dict()
    result['slices'] = []
    for slice in slices.values():
        slice_info = dict()
        slice_info['sls_number'] = slice.number
        slice_info['packet_size'] = slice.packet_size
        slice_info['bandwidth'] = slice.qos_throughput
        slice_info['qos_delay'] = slice.qos_delay
        slice_info['estimate_delay'] = slice.estimate_delay
        slice_info['flows'] = [{"lambda": flow.rho_a, 'path': flow.path} for flow in slice.flows_list]
        result['slices'].append(slice_info)

    result['topology'] = {'switches': []}
    for sw in graph.number2switch.values():
        switches_info = dict()
        switches_info['number'] = sw.number
        switches_info['ports'] = []
        for port in sw.port_list:
            port_info = dict()
            port_info['link'] = port.link
            port_info['bandwidth'] = port.physical_speed
            port_info['queues'] = [{'priority': pr.priority, 'queue_number': queue.number, 'weight': queue.weight} 
                                   for pr in port.priority_list for queue in pr.queue_list]
            switches_info['ports'].append(port_info)
        result['topology']['switches'].append(switches_info)
    
    # result['topology']['links'] = [[el[0], el[1]] for el in graph.edge2port.keys()]

    with open(f'outputs/{file_name}.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)


    


def main(argv):
    # # парсим конфиг файл и заполняем необходимы структуры
    graph, slices = parse_config(argv[0])

    beam_width = int(argv[1])
    # file_name = argv[0][11:len(argv[0])-5]

    file_name = argv[0].split("\\")[-1].split('.')[0] + f'_beam_{beam_width}'

    # сортируем слайсы в зависимости от требования к задержке
    slices_order = sorted(slices.keys(), key=lambda x: slices[x].qos_delay)
    
    main_algorithm(graph, slices, slices_order, beam_width, file_name)

    # # задаем начальные значения приоритетов и весов для виртуальных пластов на каждом коммутаторе
    # flag = set_initial_parameters(slices, slices_order, topology)
    # if flag:
    #     print('Impossible to continue calculation. Stop working')
    #     return

    # # формируем кривую обслуживания на каждом коммутаторе для начальных параметров
    # create_start_service_curve(topology)

    # # подбор корректных параметров для слайсов
    # algorithm.modify_queue_parameters(slices, slices_order, topology, file_name)

    # записываем результаты работы в выходной файл
    # file_name = argv[0].split("\\")[-1].split('.')[0] + f'_beam_{beam_width}'
    write_result(file_name, slices, graph)


if __name__ == "__main__":
    main(sys.argv[1:])