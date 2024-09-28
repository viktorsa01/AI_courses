import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=300)
import time

import pandas as pd
from PIL import Image

class Node():
    def __init__(self, coordinates, cost, reach_cost = -1, priority = 0, BestParent = None ):
        self.coordinates = coordinates
        self.cost = cost
        self.reach_cost = reach_cost
        self.priority = priority
        self.BestParent = BestParent
    
    def get_Best_Parent(self):
        return self.BestParent
    
    def set_Best_Parent(self, nParent):
        self.BestParent = nParent
    
    def set_reach_cost(self, reach_cost):
        self.reach_cost = reach_cost
    
    def set_priority(self, priority):
        self.priority = priority
    
    def get_Best_Path(self, maxIt):
        path = [self.coordinates]
        current = self.get_Best_Parent()
        it = 0
        tot_cost = self.cost
        while current != None and it <= maxIt:
            it += 1
            path.append(current.coordinates)
            tot_cost += current.cost
            current = current.get_Best_Parent()
        
        return path, tot_cost
            
class Node_Queue():
    #class for keeping track of what nodes to open first
    def __init__ (self):
        self.queue = []
        
    def empty(self):
        return len(self.queue) == 0
    
    def put(self, n):
        if self.empty():
            self.queue.append(n)
        else:
            for i in range(len(self.queue)):
                if n.priority > self.queue[i].priority:
                    self.queue.insert(i, n)
                    return
            self.queue.append(n)
            
    def pop(self):
        return self.queue.pop()
        
class Map_Obj():
    def __init__(self, task=1):
        self.start_pos, self.goal_pos, self.end_goal_pos, self.path_to_map = self.fill_critical_positions(
            task)
        self.int_map, self.str_map = self.read_map(self.path_to_map)
        self.tmp_cell_value = self.get_cell_value(self.goal_pos)
        self.set_cell_value(self.start_pos, ' S ')
        self.set_cell_value(self.goal_pos, ' G ')
        self.tick_counter = 0

    def read_map(self, path):
        """
        Reads maps specified in path from file, converts them to a numpy array and a string array. Then replaces
        specific values in the string array with predefined values more suitable for printing.
        :param path: Path to .csv maps
        :return: the integer map and string map
        """
        # Read map from provided csv file
        df = pd.read_csv(path, index_col=None,
                         header=None)  #,error_bad_lines=False)
        # Convert pandas dataframe to numpy array
        data = df.values
        # Convert numpy array to string to make it more human readable
        data_str = data.astype(str)
        # Replace numeric values with more human readable symbols
        data_str[data_str == '-1'] = ' # '
        data_str[data_str == '1'] = ' . '
        data_str[data_str == '2'] = ' , '
        data_str[data_str == '3'] = ' : '
        data_str[data_str == '4'] = ' ; '
        return data, data_str

    def fill_critical_positions(self, task):
        """
        Fills the important positions for the current task. Given the task, the path to the correct map is set, and the
        start, goal and eventual end_goal positions are set.
        :param task: The task we are currently solving
        :return: Start position, Initial goal position, End goal position, path to map for current task.
        """
        if task == 1:
            start_pos = [27, 18]
            goal_pos = [40, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_1.csv'
        elif task == 2:
            start_pos = [40, 32]
            goal_pos = [8, 5]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_1.csv'
        elif task == 3:
            start_pos = [28, 32]
            goal_pos = [6, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_2.csv'
        elif task == 4:
            start_pos = [28, 32]
            goal_pos = [6, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_Edgar_full.csv'
        elif task == 5:
            start_pos = [14, 18]
            goal_pos = [6, 36]
            end_goal_pos = [6, 7]
            path_to_map = 'Samfundet_map_2.csv'

        return start_pos, goal_pos, end_goal_pos, path_to_map

    def get_cell_value(self, pos):
        return self.int_map[pos[0], pos[1]]

    def get_goal_pos(self):
        return self.goal_pos

    def get_start_pos(self):
        return self.start_pos

    def get_end_goal_pos(self):
        return self.end_goal_pos

    def get_maps(self):
        # Return the map in both int and string format
        return self.int_map, self.str_map

    def move_goal_pos(self, pos):
        """
        Moves the goal position towards end_goal position. Moves the current goal position and replaces its previous
        position with the previous values for correct printing.
        :param pos: position to move current_goal to
        :return: nothing.
        """
        tmp_val = self.tmp_cell_value
        tmp_pos = self.goal_pos
        self.tmp_cell_value = self.get_cell_value(pos)
        self.goal_pos = [pos[0], pos[1]]
        self.replace_map_values(tmp_pos, tmp_val, self.goal_pos)

    def set_cell_value(self, pos, value, str_map=True):
        if str_map:
            self.str_map[pos[0], pos[1]] = value
        else:
            self.int_map[pos[0], pos[1]] = value

    def print_map(self, map_to_print):
        # For every column in provided map, print it
        for column in map_to_print:
            print(column)

    def pick_move(self):
        """
        A function used for moving the goal position. It moves the current goal position towards the end_goal position.
        :return: Next coordinates for the goal position.
        """
        if self.goal_pos[0] < self.end_goal_pos[0]:
            return [self.goal_pos[0] + 1, self.goal_pos[1]]
        elif self.goal_pos[0] > self.end_goal_pos[0]:
            return [self.goal_pos[0] - 1, self.goal_pos[1]]
        elif self.goal_pos[1] < self.end_goal_pos[1]:
            return [self.goal_pos[0], self.goal_pos[1] + 1]
        else:
            return [self.goal_pos[0], self.goal_pos[1] - 1]

    def replace_map_values(self, pos, value, goal_pos):
        """
        Replaces the values in the two maps at the coordinates provided with the values provided.
        :param pos: coordinates for where we want to change the values
        :param value: the value we want to change to
        :param goal_pos: The coordinate of the current goal
        :return: nothing.
        """
        if value == 1:
            str_value = ' . '
        elif value == 2:
            str_value = ' , '
        elif value == 3:
            str_value = ' : '
        elif value == 4:
            str_value = ' ; '
        else:
            str_value = str(value)
        self.int_map[pos[0]][pos[1]] = value
        self.str_map[pos[0]][pos[1]] = str_value
        self.str_map[goal_pos[0], goal_pos[1]] = ' G '

    def tick(self):
        """
        Moves the current goal position every 4th call if current goal position is not already at the end_goal position.
        :return: current goal position
        """
        # For every 4th call, actually do something
        if self.tick_counter % 4 == 0:
            # The end_goal_pos is not set
            if self.end_goal_pos is None:
                return self.goal_pos
            # The current goal is at the end_goal
            elif self.end_goal_pos == self.goal_pos:
                return self.goal_pos
            else:
                # Move current goal position
                move = self.pick_move()
                self.move_goal_pos(move)
                #print(self.goal_pos)
        self.tick_counter += 1

        return self.goal_pos

    def set_start_pos_str_marker(self, start_pos, map):
        # Attempt to set the start position on the map
        if self.int_map[start_pos[0]][start_pos[1]] == -1:
            self.print_map(self.str_map)
            print('The selected start position, ' + str(start_pos) +
                  ' is not a valid position on the current map.')
            exit()
        else:
            map[start_pos[0]][start_pos[1]] = ' S '

    def set_goal_pos_str_marker(self, goal_pos, map):
        # Attempt to set the goal position on the map
        if self.int_map[goal_pos[0]][goal_pos[1]] == -1:
            self.print_map(self.str_map)
            print('The selected goal position, ' + str(goal_pos) +
                  ' is not a valid position on the current map.')
            exit()
        else:
            map[goal_pos[0]][goal_pos[1]] = ' G '

    def show_map(self, map=None):
        """
        A function used to draw the map as an image and show it.
        :param map: map to use
        :return: nothing.
        """
        # If a map is provided, set the goal and start positions
        if map is not None:
            self.set_start_pos_str_marker(self.start_pos, map)
            self.set_goal_pos_str_marker(self.goal_pos, map)
        # If no map is provided, use string_map
        else:
            map = self.str_map

        # Define width and height of image
        width = map.shape[1]
        height = map.shape[0]
        # Define scale of the image
        scale = 20
        # Create an all-yellow image
        image = Image.new('RGB', (width * scale, height * scale),
                          (255, 255, 0))
        # Load image
        pixels = image.load()

        # Define what colors to give to different values of the string map (undefined values will remain yellow, this is
        # how the yellow path is painted)
        colors = {
            ' # ': (211, 33, 45),
            ' . ': (215, 215, 215),
            ' , ': (166, 166, 166),
            ' : ': (96, 96, 96),
            ' ; ': (36, 36, 36),
            ' S ': (255, 0, 255),
            ' G ': (0, 128, 255)
        }
        # Go through image and set pixel color for every position
        for y in range(height):
            for x in range(width):
                if map[y][x] not in colors: continue
                for i in range(scale):
                    for j in range(scale):
                        pixels[x * scale + i,
                               y * scale + j] = colors[map[y][x]]
        # Show image
        image.show()
        
    def show_path(self, path, map=None):
        """
        A function used to draw the map with the specified path as an image and show it.
        :param map: map to use, path to draw
        :return: nothing.
        """
        # If a map is provided, set the goal and start positions
        if map is not None:
            self.set_start_pos_str_marker(self.start_pos, map)
            self.set_goal_pos_str_marker(self.goal_pos, map)
        # If no map is provided, use string_map
        else:
            map = self.str_map

        # Define width and height of image
        width = map.shape[1]
        height = map.shape[0]
        # Define scale of the image
        scale = 20
        # Create an all-yellow image
        image = Image.new('RGB', (width * scale, height * scale),
                          (255, 255, 0))
        # Load image
        pixels = image.load()

        # Define what colors to give to different values of the string map (undefined values will remain yellow, this is
        # how the yellow path is painted)
        colors = {
            ' # ': (211, 33, 45),
            ' . ': (215, 215, 215),
            ' , ': (166, 166, 166),
            ' : ': (96, 96, 96),
            ' ; ': (36, 36, 36),
            ' S ': (255, 0, 255),
            ' G ': (0, 128, 255)
        }
        # Go through image and set pixel color for every position
        for y in range(height):
            for x in range(width):
                if map[y][x] not in colors: continue
                for i in range(scale):
                    for j in range(scale):
                        pixels[x * scale + i,
                               y * scale + j] = colors[map[y][x]]
        
        # Go through path and color the corresponding squares in the image yellow
        for node in path[1:-1]:
            for i in range(scale):
                    for j in range(scale):
                        pixels[node[0] * scale + i,
                               node[1] * scale + j] = (255, 255, 0)
        # Show image
        image.show()

class Node_Graph():
    def __init__ (self, map):
        #creates a grid of Nodes corresponding to map, which is a Map_Object
        init_grid = map.read_map(map.path_to_map)[0]
        self.height = len(init_grid)
        self.width = len(init_grid[0])
        self.grid = []
        for y in range (self.height):
            row = []
            for x in range(self.width):
                row.append(Node([x,y], init_grid[y][x]))    #Coordinates are flipped from initial grid, because of preference
            self.grid.append(row)  
        self.start_node = self.grid[map.get_start_pos()[0]][map.get_start_pos()[1]]  
        self.goal_node = self.grid[map.get_goal_pos()[0]][map.get_goal_pos()[1]]  
        self.start_node.set_reach_cost(0)
        
    def walkable_neighbors(self, node):
        #functions returns a list of accesible nodes from node
        neighbors = []
        x = node.coordinates[0]     
        y = node.coordinates[1]
        for i in range (3):
            if ( i!=1 and x-1+i >= 0 and x-1+i < self.width):   #from x-1 to x+1
                if self.grid[y][x-1+i].cost != -1:
                    neighbors.append(self.grid[y][x-1+i])
        for j in range (3):
            if ( j!=1 and y-1+i >= 0 and y-1+i < self.height):  #from y-1 to y+1
                if self.grid[y-1+j][x].cost != -1:
                    neighbors.append(self.grid[y-1+j][x])            
        return neighbors
        

def h(start, end):
    #heruistic function calculates Manhattan distance between nodes start and end
    x_0 = start.coordinates[0]
    y_0 = start.coordinates[1]
    x_1 = end.coordinates[0]
    y_1 = end.coordinates[1]
    return abs(x_1-x_0) + abs(y_1-y_0)


def A_star(map):   
    graph = Node_Graph(map)  
    open = Node_Queue()     #Keeps track of reachable nodes
    open.put(graph.start_node)   
    it = 0
    
    while not open.empty() and it < 10000:
        it+=1
        current = open.pop()    #algorithm pops the most promising node from open
        
        if current == graph.goal_node :
            return current.get_Best_Path(it), it
        
        for next in graph.walkable_neighbors(current):  #Checks all neighbors
            new_cost = current.reach_cost + next.cost   #Finds total cost to next traveling through current
            if next.reach_cost == -1 or new_cost < next.reach_cost:     #If node is not reached before, or the current path is cheaper, update node values
                next.set_reach_cost(new_cost)
                next.set_priority(new_cost + h(next, graph.goal_node))
                open.put(next)
                next.set_Best_Parent(current)
               
         
    return []   #There was no possible path or the algorithm used to many iterations


#task 1 
map = Map_Obj(task=1)
result = A_star(map)
shortest_path = result[0][0]
total_cost = result[0][1]
iterations = result[1]
map.show_path(shortest_path)
print("Task 1: ")
print("The algorithm took " + str(iterations) + " iterations.")
print("The cost of the shortest path was " + str(total_cost) + ".")

#task 2 
map = Map_Obj(task=2)
result = A_star(map)
shortest_path = result[0][0]
total_cost = result[0][1]
iterations = result[1]
map.show_path(shortest_path)
print("Task 2: ")
print("The algorithm took " + str(iterations) + " iterations.")
print("The cost of the shortest path was " + str(total_cost) + ".")

#task 3 
map = Map_Obj(task=3)
result = A_star(map)
shortest_path = result[0][0]
total_cost = result[0][1]
iterations = result[1]
map.show_path(shortest_path)
print("Task 3: ")
print("The algorithm took " + str(iterations) + " iterations.")
print("The cost of the shortest path was " + str(total_cost) + ".")

#task 4 
map = Map_Obj(task=4)
result = A_star(map)
shortest_path = result[0][0]
total_cost = result[0][1]
iterations = result[1]
map.show_path(shortest_path)
print("Task 4: ")
print("The algorithm took " + str(iterations) + " iterations.")
print("The cost of the shortest path was " + str(total_cost) + ".")
