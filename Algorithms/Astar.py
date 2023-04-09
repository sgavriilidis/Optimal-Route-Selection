from cmath import inf
import copy
import time
import sys
from abc import ABC
from turtle import distance
from typing import Tuple, Union, Dict, List, Any
import math
from matplotlib.pyplot import close
import numpy as np
 
from commonroad.scenario.trajectory import State
 
sys.path.append('../')
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import Node, CostNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.queue import FIFOQueue, LIFOQueue
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization
 
class SequentialSearch(SearchBaseClass, ABC):
    """
    Abstract class for search motion planners.
    """
 
    # declaration of class variables
    path_fig: Union[str, None]
 
    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
 
    def initialize_search(self, time_pause, cost=True):
        """
        initializes the visualizer
        returns the initial node
        """
        self.list_status_nodes = []
        self.dict_node_status: Dict[int, Tuple] = {}
        self.time_pause = time_pause
        self.visited_nodes = []
 
        # first node
        if cost:
            node_initial = CostNode(list_paths=[[self.state_initial]],
                                        list_primitives=[self.motion_primitive_initial],
                                        depth_tree=0, cost=0)
        else:
            node_initial = Node(list_paths=[[self.state_initial]],
                                list_primitives=[self.motion_primitive_initial],
                                depth_tree=0)
        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem,
                              self.config_plot, self.path_fig)
        self.dict_node_status = update_visualization(primitive=node_initial.list_paths[-1],
                                                     status=MotionPrimitiveStatus.IN_FRONTIER,
                                                     dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(self.list_status_nodes), time_pause=self.time_pause)
        self.list_status_nodes.append(copy.copy(self.dict_node_status))
        return node_initial
 
    def take_step(self, successor, node_current, cost=True):
        """
        Visualizes the step of a successor and checks if it collides with either an obstacle or a boundary
        cost is equal to the cost function up until this node
        Returns collision boolean and the child node if it does not collide
        """
        # translate and rotate motion primitive to current position
        list_primitives_current = copy.copy(node_current.list_primitives)
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        list_primitives_current.append(successor)
        self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
        if cost:
            child = CostNode(list_paths=self.path_new,
                                 list_primitives=list_primitives_current,
                                 depth_tree=node_current.depth_tree + 1,
                                 cost=self.cost_function(node_current))
        else:
            child = Node(list_paths=self.path_new, list_primitives=list_primitives_current,
                         depth_tree=node_current.depth_tree + 1)
 
        # check for collision, skip if is not collision-free
        if not self.is_collision_free(path_translated):
 
            position = self.path_new[-1][-1].position.tolist()
            self.list_status_nodes, self.dict_node_status, self.visited_nodes = self.plot_colliding_primitives(current_node=node_current,
                                                                                           path_translated=path_translated,
                                                                                           node_status=self.dict_node_status,
                                                                                           list_states_nodes=self.list_status_nodes,
                                                                                           time_pause=self.time_pause,
                                                                                           visited_nodes=self.visited_nodes)
            return True, child
        self.update_visuals()
        return False, child
 
    def update_visuals(self):
        """
        Visualizes a step on plot
        """
        position = self.path_new[-1][-1].position.tolist()
        if position not in self.visited_nodes:
            self.dict_node_status = update_visualization(primitive=self.path_new[-1],
                                                         status=MotionPrimitiveStatus.IN_FRONTIER,
                                                         dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(self.list_status_nodes), time_pause=self.time_pause)
            self.list_status_nodes.append(copy.copy(self.dict_node_status))
        self.visited_nodes.append(position)
 
    def goal_reached(self, successor, node_current):
        """
        Checks if the goal is reached.
        Returns True/False if goal is reached
        """
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        # goal test
        if self.reached_goal(path_translated):
            # goal reached
            self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
            path_solution = self.remove_states_behind_goal(self.path_new)
            self.list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=self.dict_node_status,
                                                        list_states_nodes=self.list_status_nodes, time_pause=self.time_pause)
            return True
        return False
 
    def get_obstacles_information(self):
        """
        Information regarding the obstacles.
        Returns a list of obstacles' information, each element
        contains information regarding an obstacle:
        [x_center_position, y_center_position, length, width]
 
        """
        return self.extract_collision_obstacles_information()
 
    def get_goal_information(self):
        """
        Information regarding the goal.
        Returns a list of the goal's information
        with the following form:
        [x_center_position, y_center_position, length, width]
        """
        return self.extract_goal_information()
 
    def get_node_information(self, node_current):
        """
        Information regarding the input node_current.
        Returns a list of the node's information
        with the following form:
        [x_center_position, y_center_position]
        """
        return node_current.get_position()
 
    def get_node_path(self, node_current):
        """
        Information regarding the input node_current.
        Returns the path starting from the initial node and ending at node_current.
        """
        return node_current.get_path()
 
    def cost_function(self, node_current):
        """
        Returns g(n) from initial to current node, !only works with cost nodes!
        """
        velocity = node_current.list_paths[-1][-1].velocity
 
        node_center = self.get_node_information(node_current)
        goal_center = self.get_goal_information()
        distance_x = goal_center[0] - node_center[0]
        distance_y = goal_center[1] - node_center[1]
        length_goal = goal_center[2]
        width_goal = goal_center[3]
 
        distance = 4.5
        if(abs(distance_x)<length_goal/2 and abs(distance_y)<width_goal/2):
            prev_x = node_current.list_paths[-2][-1].position[0]
            prev_y = node_current.list_paths[-2][-1].position[1]
            distance = goal_center[0] - length_goal / 2 - prev_x
        cost = node_current.cost + distance
       
        return cost
 
    def heuristic_function(self, node_current):
        """
        Enter your heuristic function h(x) calculation of distance from node_current to goal
        Returns the distance normalized to be comparable with cost function measurements
        """
        n=1
        dis=float('inf')
        wo=0

        for obstacles in self.get_obstacles_information():
            obstacle_info = (obstacles[0],obstacles[1])
            if dis > self.distance(node_current.get_position(),obstacle_info,0):
                dis=self.distance(node_current.get_position(),obstacle_info,0)
                obstacl=obstacles
                #print(*obstacle_info)
             
    
        if obstacl[1]< 0:
            obstacle_info = (obstacl[0],obstacl[1])
            wo = self.distance(node_current.get_position(),obstacl,n)
            
        else:
            obstacle_info = (obstacl[0],obstacl[1] )
            wo = self.distance(node_current.get_position(),obstacl,n)

        nodeinfo= self.get_node_information(node_current)
        g_w=(30/wo)
        if obstacle_info[0] >  nodeinfo[0] :
            distance =self.distance(node_current.get_position(),self.get_goal_information(),n) + g_w 
            #print("me baros" , distance, "=" , self.distance(node_current.get_position(),self.get_goal_information(),1), "*" , g_w)
        else:
           
            distance = self.distance(node_current.get_position(),self.get_goal_information(),n) 
            #print("xwris" , distance , "=" , self.distance(node_current.get_position(),self.get_goal_information(),1), "*" , g_w )

        
        #distance = self.distance(node_current.get_position(),self.get_goal_information(),n) 
        #distance=0
        return distance



    def evaluation_function(self, node_current):
        """
        f(x) = g(x) + h(x)
        """
        g = self.cost_function(node_current)
        h = self.heuristic_function(node_current)
        f = g + h*5 
        #print("to baros einai :", f , "=" , g , "+" , h)  
        return f

 
    def Astar_search(self,node_current):
        open_node = []
        close_node = []
        open_node.append(node_current)
        min_cost = float('inf')
        print(*self.get_obstacles_information()) 
        num_of_nodes = 1 
        while len(open_node)>0 :
 
            for op in open_node:
                #print(self.get_node_information(op), "op")
                #print(self.evaluation_function(op), " eva")
                if self.evaluation_function(op)< min_cost :  ####
                    node_current=op
                    min_cost=self.evaluation_function(op)
 
            num_of_nodes +=1
            close_node.append(node_current)
            #print(self.get_node_information(close_node[0]))
            open_node.remove(node_current)   

            #sucks=node_current.get_successors()
            #print(*sucks,"sucks")
 
            for primitive_successor in node_current.get_successors():

                #print (i+1)
                
                if primitive_successor in close_node:
                    continue
                "if primitive_successor not in open_node:"

                #if primitive_successor not in open_node and primitive_successor not in close_node :
                
                collision_flag ,child = self.take_step(successor=primitive_successor, node_current=node_current)
                if collision_flag:
                    continue    
                open_node.append(child)   
                #print(self.get_node_information(child), "primitive succesor")

                goal_flag = self.goal_reached(successor=primitive_successor, node_current=node_current)
                # if goal is reached, return back with the solution path
                if goal_flag:
                    return child,num_of_nodes    
 
           

            min_cost = float('inf')
        return False,num_of_nodes
 
    def execute_search(self, time_pause) -> Tuple[Union[None, List[List[State]]], Union[None, List[MotionPrimitive]], Any]:
        node_initial = self.initialize_search(time_pause=time_pause)
        print(self.get_obstacles_information())
        print(self.get_goal_information())
        print(self.get_node_information(node_initial))
        """Enter your code here"""
        final_child,visited_nodes = self.Astar_search(node_current=node_initial)
        found_path = final_child.get_path()

        f = open('results.txt','a')
        f.write('< Scenario 1 > \n')
        f.write('A* (w=1): \n')
        f.write('\t Visited Nodes number: '+ str(visited_nodes) +'\n')
        f.write('\t Path: ')
        for node_in_Path in found_path:
            f.write('('+ str(node_in_Path[0]) +',' + str(node_in_Path[1]) +')')
            #if(node_in_Path.all() != found_path[-1].all()):
            f.write('->') 
        f.write('\n') 
        f.write('\t Heuristic Cost (initial node): '+ str(self.heuristic_function(node_initial)) +'\n')   
        f.write('\t Estimated Cost: ' + str(self.cost_function(final_child)) +'\n' ) 


        return found_path
 
class Astar(SequentialSearch):
    """
    Class for Astar Search algorithm.
    """
 
    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
 
 
 


