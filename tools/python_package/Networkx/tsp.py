import sys
# sys.path.append("../")
# sys.path.append("../../")
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians,sin,cos,asin,sqrt
import math
import time
import json
import random
# from utils.TimeUtil import get_time_str,get_cur_time
# from utils.LogUtil import print_info
# from utils.Utils import get_host_by_nacos,haversine_dis,get_plan_coordinate,get_dis_cost,fill_by_city_info,get_config_value
from math import radians,sin,cos,asin,sqrt
import requests,warnings
import math
from scipy import spatial
import copy
import traceback
import numpy as np
warnings.filterwarnings("ignore")

EARTH_RADIUS = 6371393.0
RADIUS_SCALE = math.pi / 180.

def haversine_dis(lon1, lat1, lon2, lat2):
    #将十进制转为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    #haversine公式
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    aa = sin(d_lat/2)**2 + cos(lat1)*cos(lat2)*sin(d_lon/2)**2
    c = 2 * asin(sqrt(aa))
    return c*EARTH_RADIUS

class TSPSolver(object):
    def __init__(self,worker_lon,worker_lat, station_list):
        self.worker_lon = worker_lon
        self.worker_lat = worker_lat
        self.station_list = station_list
   

    def create_data_model(self):
        station_list = self.station_list

        """Stores the data for the problem."""
        data = {}
        data['point'] = [[self.worker_lon,self.worker_lat]]

        for s in station_list:
            data['point'].append([ s["lon"], s["lat"]])

        distances = []
        for sp in data['point']:
            row_list = [0]
            for ep in data['point'][1:]:
                row_list.append(  int(haversine_dis(sp[0], sp[1],ep[0], ep[1] )))
            distances.append(row_list)
        
        data['distance_matrix'] = distances
        data['num_vehicles'] = 1
        data['depot'] = 0                        
        self.data = data
        return data

    def print_solution(self):
        data, manager, routing, solution = self.data,self.manager,self.routing,self.solution
        """Prints solution on console."""
        print('Objective: {} miles'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = 'Route for vehicle 0:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        print(plan_output)
        plan_output += 'Route distance: {}miles\n'.format(route_distance)

    def get_solution(self):
        data, manager, routing, solution = self.data,self.manager,self.routing,self.solution
        """Prints solution on console."""
    
        #print(f'Objective: {solution.ObjectiveValue()}')
        res = {
            #"station_list":self.station_list,
              "total_distance":solution.ObjectiveValue(), "solve_time":self.solve_time
              }
        """Prints solution on console."""
        #print('Objective: {} miles'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = 'Route for vehicle 0:\n'
        route_distance = 0
        routes = []
        routes_idx = []

        route_distances = []
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            
            station_idx = manager.IndexToNode(index)-1
            routes_idx.append(station_idx)
            route_distances.append(route_distance)         
        
        for order,idx in enumerate(routes_idx[:-1]):
            station = self.station_list[idx]
            station["tsp_order"] = order
            station["route_distance"] = route_distances[order]
            routes.append(station)
            
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        #print(plan_output,routes_idx)
        res["routes"] = routes
        plan_output += 'Route distance: {}miles\n'.format(route_distance)
        #print('Total distance of all routes: {}m'.format(total_distance))
        #print('Total load of all routes: {}'.format(total_load))
        return res
    
    def solve(self):
        # Instantiate the data problem.
        data = self.create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    #     search_parameters.local_search_metaheuristic = (
    #         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(1)
        
        # Solve the problem.
        ts = time.time()
        solution = routing.SolveWithParameters(search_parameters)
        self.solve_time = time.time() - ts            
        
        self.data = data
        self.manager = manager
        self.routing = routing
        self.solution = solution
        
                # Print solution on console.
#         if solution:
#             self.print_solution()
            
        return self.get_solution()