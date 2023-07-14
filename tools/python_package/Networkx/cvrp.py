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

class CVRPSolver(object):
    def __init__(self,station_list,bike_list):
        self.station_list = station_list
        self.bike_list = bike_list
        
    def create_data_model(self):
        station_list = self.station_list
        bike_list = self.bike_list
        """Stores the data for the problem."""
        data = {}
        data['point'] = []
        data['demands']=[]
        data['vehicle_capacities']=[]

        for s in station_list:
            data['point'].append([ s["lon"], s["lat"]])
            data['demands'].append(0)
            data['vehicle_capacities'].append(int(s["move_num"]))

        data['num_vehicles']=len(data['vehicle_capacities'])
        data['starts']=data['ends']=[x for x  in range(len(data['vehicle_capacities']))]

        for b in bike_list:
            data['point'].append([ b["longitude"], b["latitude"]])
            data['demands'].append(int(1))
        
        distances = []
        for sp in data['point']:
            row_list = []
            for ep in data['point']:
                row_list.append(int(haversine_dis(sp[0], sp[1],ep[0], ep[1] )))
            distances.append(row_list)
        data['distance_matrix'] = distances
        
        #print(f"sum demands {sum(data['demands'])} sum(capacities) {sum(data['vehicle_capacities'])}" )
        return data

    def print_solution(self):
        data, manager, routing, solution = self.data,self.manager,self.routing,self.solution
        """Prints solution on console."""
        print(f'Objective: {solution.ObjectiveValue()}')
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                     route_load)
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            plan_output += 'Load of the route: {}\n'.format(route_load)
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print('Total distance of all routes: {}m'.format(total_distance))
        print('Total load of all routes: {}'.format(total_load))

    def get_solution(self):
        data, manager, routing, solution = self.data,self.manager,self.routing,self.solution
        """Prints solution on console."""
    
        #print(f'Objective: {solution.ObjectiveValue()}')
#         dropped_nodes = 'Dropped nodes:'
#         for node in range(routing.Size()):
#             if routing.IsStart(node) or routing.IsEnd(node):
#                 continue
#             if solution.Value(routing.NextVar(node)) == node:
#                 dropped_nodes += ' {}'.format(manager.IndexToNode(node))
#         #print(dropped_nodes)
        
        res = {"station_list":self.station_list, "bike_list":self.bike_list 
              ,"total_distance":solution.ObjectiveValue(), "solve_time":self.solve_time
              }
        total_distance = 0
        total_load = 0
        routes = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
            route_distance = 0
            route_load = 0
            bikes_idx = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
                bikes_idx.append(manager.IndexToNode(index))
            
            bikes = [self.bike_list[b - len(self.station_list)] for b in bikes_idx[:-1]]
                
            plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                     route_load)
            
            plan_output += 'Distance of the route: {}m\n'.format(route_distance)
            plan_output += 'Load of the route: {}\n'.format(route_load)
            #print(plan_output)
            #print(vehicle_id,bikes_idx[:-1],[b - len(self.station_list) for b in bikes_idx[:-1]])
            self.station_list[vehicle_id]["route_distance"] = route_distance
            self.station_list[vehicle_id]["route_load"] = route_load
            self.station_list[vehicle_id]["bikes"] = bikes
            routes.append({
                "station":self.station_list[vehicle_id]
                         })
            
            
            total_distance += route_distance
            total_load += route_load
        res["routes"] = routes
        #print('Total distance of all routes: {}m'.format(total_distance))
        #print('Total load of all routes: {}'.format(total_load))
        return res
    
    def solve(self):
        """Solve the CVRP problem."""
        # Instantiate the data problem.
        
        data = self.create_data_model()

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['point']),
                                               data['num_vehicles'], data['starts'], data['ends'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        # Create and register a transit callback.
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
#             distance = haversine_dis(data['point'][from_node][0], data['point'][from_node][1],data['point'][to_node][0], data['point'][to_node][1])
#             return distance

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')
        
        penalty = 10000000 
        for node in range(data['num_vehicles'], len(data['distance_matrix'])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
#         search_parameters.local_search_metaheuristic = (
#             routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.FromSeconds(1)

        # Solve the problem.
        ts = time.time()
        solution = routing.SolveWithParameters(search_parameters)
        self.solve_time = time.time() - ts
        
        self.data = data
        self.manager = manager
        self.routing = routing
        self.solution = solution
        return self.get_solution()