
   
import math
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from tabulate import tabulate
from collections import defaultdict
import random



import heapq #library to use priority queue for shortest path algorithm

class Statenode: # class statenode is a class for the details of the state vertex
    def __init__(self, S_name, population, infected):
        self.StateName = S_name
        self.population = population
        self.infected_no = infected
        self.cityList = [] #list for heap representation
        self.cityList1 = [] #list for grapg representation of the cities in a state
        self.medCityList = []
        self.neighbourState = [] # edge list representation of the neighbour vertices
        self.mostInfected = None
        self.avlSupplies  = 0
        self.density=infected/population


class cityNode: #class city node is a class for the details of the city vertex
    def __init__(self, state, C_name, population, infected,med=False,capacity=5000):
        self.stateName = state
        self.cityName = C_name
        self.population = population
        self.capacity=capacity
        self.infected_no = infected
        self.distance = float('inf') #initially set to infinity
        self.parentCity = None
        self.medCity = med #only if its a medical city,this will be true
        self.neighbourCity = [] # edge list for city vertex
        self.supplies = 0

    def __lt__(self, other):
        return self.distance < other.distance

class edgeDist: 
#class to create edges in graphs of states and cities, it has source,destination and weight of the edge #direction can be ignored for now
    def __init__(self, source, destination, distance, direction):
        self.source = source
        self.destination = destination
        self.distance = distance
        self.direction = direction


class StateGraph:
    #class for the whole graph representation of the states
    def __init__(self):
        self.stateVertices = [] #the list for all states

    #this function adds new state to the grapg, initially unconnected  with any other states
    def addNewState(self, Sname, population, infected):
        newState = Statenode(Sname, population, infected)
        self.stateVertices.append(newState)

    # this adds an edge between 2 states that is present in the graph
    def addNeighbourState(self, state1, state2, dist, direction):
        state1_vertex = next((state for state in self.stateVertices if state.StateName == state1), None) #checks if state1 is in the graph if not returns None
        state2_vertex = next((state for state in self.stateVertices if state.StateName == state2), None) #checks if state2 is in the graph

        if state1_vertex and state2_vertex:
            state1_vertex.neighbourState.append(edgeDist(state1, state2, dist, direction)) #add the states and weights to its respective edge list 
            state2_vertex.neighbourState.append(edgeDist(state2, state1, dist, direction))

    # when a city is added to city graph, it is then added to its respective state list
    def addCitiesToStates(self, cityVertex):
        for stateVertex in self.stateVertices:
            if stateVertex.StateName == cityVertex.stateName:
                stateVertex.cityList.append(cityVertex) #add to the heap representatioin list
                stateVertex.cityList1.append(cityVertex) #add to the graph representation list
                if cityVertex.medCity:
                    stateVertex.medCityList.append(cityVertex) #if its a medical city, its added to that list

    # when the cities are added to state , its then converted into a heap in the cityList, as we add a city, this function is called.
    def reOrderCities(self):
        for stateVertex in self.stateVertices:
            stateVertex.cityList = self.buildCityHeap(stateVertex.cityList, len(stateVertex.cityList)) #build city heap is called
            stateVertex.mostInfected = stateVertex.cityList[0] if stateVertex.cityList else None #set the most infected as root of the maxheap

    #for the given state, find number of med city and provide the medical supplies equally
    def supplyState(self, units, stateNamee):
        count = 0
        # Calculate the total capacity of medical cities in the state
        for stateVertex in self.stateVertices:
            if stateVertex.StateName == stateNamee:
                stateVertex.avlSupplies += units
                count = len(stateVertex.medCityList)
                state = stateVertex
                break

        # Calculate the count of medical cities in the state

        if count == 0:
            print("No medical cities found in the state.")
            return

        # Distribute supplies based on capacity proportionally
        for city in state.medCityList:
            city.supplies += units//count
            print("City", city.cityName, "in state", state.StateName, "has been provided supply of",
                units//count, "units")

        state.avlSupplies -= units  # Deduct the allocated supplies from available supplies



    #this function restores the max heap property of the cityList
    def maxHeapify(self, cityList, N, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < N and cityList[l].infected_no > cityList[largest].infected_no: #if left child is largest,its refered
            largest = l
        if r < N and cityList[r].infected_no > cityList[largest].infected_no:#if right  child is largest, its swapped
            largest = r
        if largest != i:
            cityList[i], cityList[largest] = cityList[largest], cityList[i] # if the largest has changed, then its swapped in main list
            self.maxHeapify(cityList, N, largest)

    #function to build the heap using reverse level order traversal
    def buildCityHeap(self, cityList, N):
        semi_root = N // 2
        #traversing only one half of the list
        for i in range(semi_root - 1, -1, -1):
            self.maxHeapify(cityList, N, i)
        return cityList

    #to provide medical assitance to most infected state, the most infected state from the max heap is removed. then the path to that is found.
    def serviceCityInState(self, stateName):
        state = next((s for s in self.stateVertices if s.StateName == stateName), None)
        if state and state.cityList:
            mostInfectedCity = state.cityList[0] #max element according to infected rate
            state.cityList[0].infected_no -= 500 # decrease the number of infected after servicing a city
            print("Serviced:", mostInfectedCity.cityName)
            self.moveToCityShortestPath(mostInfectedCity, state) # function to find the math to medical city
            state.cityList[0], state.cityList[-1] = state.cityList[-1], state.cityList[0]# first element is swapped with last
            self.maxHeapify(state.cityList, len(state.cityList), 0) #restore heap property
            state.mostInfected = state.cityList[0] if state.cityList else None
            
    def calculateStateInfectionRate(stateGraph, stateName):
        state = None
        for s in stateGraph.stateVertices:
            if s.StateName == stateName:
                state = s
                break

        if state:
            return (state.infected_no / state.population) * 100
        return None 
    
#this function generates a report for a given state name by showing all its details in terminal in tabular format
#tabulate module is used - O(n){To find the required state}
    def generatereport(self,statename):
        curr=None
        for i in self.stateVertices:
            if i.StateName==statename:
                curr=i
                break
        if curr is None:
            print("City not found")
        state_data = [
        ["State Name", curr.StateName],
        ["Population", curr.population],
        ["Infected No", curr.infected_no],
        ["Density", curr.density],
        ["Most Infected City", curr.mostInfected.cityName if curr.mostInfected else "None"]
        ]
        
        print(tabulate(state_data, headers=["Attribute", "Value"], tablefmt="grid"))

#this prints the the state which has the maximum affected density by calculating it
# O(n) - To iterate through all the states
    def getmaxstate(self):
        max=self.stateVertices[0]
        for i in self.stateVertices:
            if max.density < i.density:
                max=i
        print("The State with the maximum affected people density is: ",max.StateName)       
    
    '''This function is used to find the shortest path from the infected city to a medical city. 
    we first find the shortest path from the max node then travel in reverse hypothetically.
    used dijkstra's shortest path algorithm.'''
    
    def moveToCityShortestPath(self, city, stateVer):
        for ver in stateVer.cityList1:  # initialize the distances
            ver.distance = float('inf')  # infinity
            ver.parentCity = None
        city.distance = 0
        priority_queue = []  # using priority queue
        heapq.heappush(priority_queue, (city.distance, city))  # pushing city with its distance

        while priority_queue:
            current_distance, current_city = heapq.heappop(priority_queue)  # extract the city with the smallest distance

            if current_city.medCity:
                print(f"Reached medical city: {current_city.cityName} with total distance {current_distance}")
                return current_distance  # if a med city is reached, exit the function

            for edge in current_city.neighbourCity:
                neighbor_city = edge.destination
                distance = edge.distance
                new_distance = current_distance + distance 

                if new_distance < neighbor_city.distance:  # relaxation part
                    neighbor_city.distance = new_distance
                    neighbor_city.parentCity = current_city
                    heapq.heappush(priority_queue, (new_distance, neighbor_city))  # push the updated distance and city

        print("No medical city reachable")
        return float('inf')
    

    def reachAllCities(self, state_name):
       #This function reachAllCities is a method of a class. It takes one parameter, state_name, which specifies the name of the state whose cities we want to connect using a Minimum Spanning Tree (MST).
        state = next((s for s in self.stateVertices if s.StateName == state_name), None)
        #This line searches through self.stateVertices to find the state object that matches the state_name parameter. If no match is found, state will be None.
#If the state is not found, the function returns an empty list, indicating that no MST can be constructed.
       if not state:
            return []
#This selects the first city from the state's medCityList to use as the starting node for the algorithm.
        start_node = state.medCityList[0]
        graph = defaultdict(dict)
#This creates a dictionary state_cities where the keys are city names and the values are city objects, but only for cities within the specified state.
        state_cities = {city.cityName: city for city in state.cityList}
#It checks if the neighbor city is within the state_cities dictionary to ensure only intra-state connections are included.
        for city in state.cityList:
            for neighbor in city.neighbourCity:
                if neighbor.destination.cityName in state_cities:
                    graph[city.cityName][neighbor.destination.cityName] = neighbor.distance

        mst = []
        visited = set([start_node.cityName])
        pq = []

    # Push the edges of the start node into the priority queue
        for neighbor in start_node.neighbourCity:
            if neighbor.destination.cityName in state_cities:
                heapq.heappush(pq, (neighbor.distance, start_node.cityName, neighbor.destination.cityName))
#This loop pushes the edges of the start_node into the priority queue pq with the distance as the priority. This ensures that we start with the smallest edge from the starting node.


#This loop runs until all cities in state_cities have been visited.
#This nested loop extracts the minimum weight edge from the priority queue. If the destination city (dest) is not already visited, the loop breaks, and the edge is processed.
#The else part of the loop ensures that if the priority queue becomes empty and no valid edge is found, the outer loop breaks.
        while len(visited) < len(state_cities):
            while pq:
                weight, source, dest = heapq.heappop(pq)
                if dest not in visited:
                 break
            else:
                break

            mst.append((source, dest, weight))
            visited.add(dest)
#This loop pushes the edges of the newly visited city (dest) into the priority queue, ensuring that only edges to unvisited cities are considered.
            for neighbor in state_cities[dest].neighbourCity:
                if neighbor.destination.cityName not in visited and neighbor.destination.cityName in state_cities:
                    heapq.heappush(pq, (neighbor.distance, dest, neighbor.destination.cityName))


        return mst


#this class is an ADT implementation of heap library, this is not used currently, changes are to be made
class priorityQueue:
    def __init__(self, citylist):
        self.queueMin = citylist
        self.buildMinPQ(citylist)

    def minHeapify(self, cityListQueue, n, i):
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n and cityListQueue[l] is not None and cityListQueue[l].distance < cityListQueue[i].distance:
            small = l
        else:
            small = i
        if r < n and cityListQueue[r] is not None and cityListQueue[r].distance < cityListQueue[small].distance:
            small = r
        if small != i:
            cityListQueue[i], cityListQueue[small] = cityListQueue[small], cityListQueue[i]
            self.minHeapify(cityListQueue, n, small)

    def buildMinPQ(self):
        n = len(self.queueMin)
        strt = (n // 2) - 1
        for i in range(strt, -1, -1):
            self.minHeapify(self.queueMin, n, i)
        return self.buildMinPQ

    def extract_Min(self):
        n = len(self.queueMin)
        print("deleted:", self.queueMin[0].distance)
        extractMin = self.queueMin[0]
        self.queueMin[0], self.queueMin[n-1] = self.queueMin[n-1], self.queueMin[0]
        self.queueMin.pop()
        self.minHeapify(self.queueMin, n-1, 0)
        return extractMin

    def decrease_Key(self, v, v_d):
        for i in self.queueMin:
            if i == v:
                i.distance = v_d
                self.minHeapify(self.queueMin, len(self.queueMin), 0)

#this class is to create the city grapg
class CityGraph:
    def __init__(self):
        self.cityVertices = []

    #add a new city vertex to the graph
    def addCity(self, Sname, Cname, population, infected, stateGraph,med = False):
        temp_city = cityNode(Sname, Cname, population, infected,med)
        self.cityVertices.append(temp_city)
        stateGraph.addCitiesToStates(temp_city)

    #add the edges to the city, edge list representation
    def connectCities(self, city1, city2, distance):
        tempCity1 = next((city for city in self.cityVertices if city.cityName == city1), None)
        tempCity2 = next((city for city in self.cityVertices if city.cityName == city2), None)

        if tempCity1 and tempCity2:
            tempCity1.neighbourCity.append(edgeDist(tempCity1, tempCity2, distance, None))
            tempCity2.neighbourCity.append(edgeDist(tempCity2, tempCity1, distance, None))
    
    def getcity(self, name):
        for city in self.cityVertices:
            if city.cityName == name:
                return city
        
        print("City not found!!")
        return None

'''This evacuates the city provided by moving the population in the city to first nearby city upto their 
capacity and if it gets filles then moves to the next nearest city
For sorting - O(nlogn)
For iterating - O(n)
'''

    def evacuatecity(self, name):
        curr = self.getcity(name)
        
        population = curr.population
        neighbours = curr.neighbourCity
        neighbours = sorted(neighbours, key=lambda node: node.distance)
        
        for i in neighbours:
            new=self.getcity(i.destination.cityName)
            if population <= 0:
                break
            
            available_capacity = new.capacity - new.population
            
            if available_capacity > 0:
                if population <= available_capacity:
                    new.population += population
                    population = 0
                else:
                    new.population += available_capacity
                    population -= available_capacity
        
        # Optionally, print the updated populations
        print(f"After evacuation, the population of {name} is now 0.")
        curr.population = 0  # Set the evacuated city's population to 0
        for i in neighbours:
            new=self.getcity(i.destination.cityName)
            print(f"{new.cityName} now has a population of {new.population}.")

    def getNearestMedicalCity(cityGraph, cityName):
        city = None
        for c in cityGraph.cityVertices:
            if c.cityName == cityName:
                city = c
                break

        if city:
            for c in cityGraph.cityVertices:
                c.distance = float('inf')
            city.distance = 0
            priority_queue = [(city.distance, city)]
            heapq.heapify(priority_queue)

            while priority_queue:
                current_distance, current_city = heapq.heappop(priority_queue)
                if current_city.medCity:
                    return current_city
                for edge in current_city.neighbourCity:
                    neighbor_city = edge.destination
                    distance = edge.distance
                    new_distance = current_distance + distance

                    if new_distance < neighbor_city.distance:
                        neighbor_city.distance = new_distance
                        heapq.heappush(priority_queue, (new_distance, neighbor_city))
            return None
        return None
    
    def findBridges(self):
        self.time = 0
        visited = {city.cityName: False for city in self.cityVertices}
        disc = {city.cityName: float('inf') for city in self.cityVertices}
        low = {city.cityName: float('inf') for city in self.cityVertices}
        parent = {city.cityName: None for city in self.cityVertices}
        bridges = []

        for city in self.cityVertices:
            if not visited[city.cityName]:
                self.bridgeUtil(city, visited, disc, low, parent, bridges)

        return bridges

    def bridgeUtil(self, u, visited, disc, low, parent, bridges):
        visited[u.cityName] = True
        disc[u.cityName] = self.time
        low[u.cityName] = self.time
        self.time += 1

        for edge in u.neighbourCity:
            v = edge.destination
            if not visited[v.cityName]:
                parent[v.cityName] = u.cityName
                self.bridgeUtil(v, visited, disc, low, parent, bridges)

                low[u.cityName] = min(low[u.cityName], low[v.cityName])

                if low[v.cityName] > disc[u.cityName]:
                    bridges.append((u.cityName, v.cityName))

            elif v.cityName != parent[u.cityName]:
                low[u.cityName] = min(low[u.cityName], disc[v.cityName])
                
    
    def findArticulationPoints(self):
        self.time = 0
        visited = {city.cityName: False for city in self.cityVertices}
        disc = {city.cityName: float('inf') for city in self.cityVertices}
        low = {city.cityName: float('inf') for city in self.cityVertices}
        parent = {city.cityName: None for city in self.cityVertices}
        articulation_points = {city.cityName: False for city in self.cityVertices}

        for city in self.cityVertices:
            if not visited[city.cityName]:
                self.articulationPointUtil(city, visited, disc, low, parent, articulation_points)

        return [city for city, is_ap in articulation_points.items() if is_ap]

    def articulationPointUtil(self, u, visited, disc, low, parent, articulation_points):
        children = 0
        visited[u.cityName] = True
        disc[u.cityName] = self.time
        low[u.cityName] = self.time
        self.time += 1

        for edge in u.neighbourCity:
            v = edge.destination
            if not visited[v.cityName]:
                parent[v.cityName] = u.cityName
                children += 1
                self.articulationPointUtil(v, visited, disc, low, parent, articulation_points)

                low[u.cityName] = min(low[u.cityName], low[v.cityName])

                if parent[u.cityName] is None and children > 1:
                    articulation_points[u.cityName] = True

                if parent[u.cityName] is not None and low[v.cityName] >= disc[u.cityName]:
                    articulation_points[u.cityName] = True

            elif v.cityName != parent[u.cityName]:
                low[u.cityName] = min(low[u.cityName], disc[v.cityName])
                
                
    

def visualisation(state_graph):
    """
    Visualize the graph using the networkx module such that states are connected and their respective distances are shown in between.
    Cities must be in the form of a heap, and for each vertex in the state, there must be a separate heap data structure.
    """
    state_interconnections = nx.DiGraph()

    # Create graphs for individual states
    state_graphs = {}
    combined_city_graph = nx.DiGraph()

    for state in state_graph.stateVertices:
        state_name = state.StateName
        state_graphs[state_name] = nx.DiGraph()

        # Add state nodes and edges
        for city in state.cityList:
            state_graphs[state_name].add_node(city.cityName, population=city.population, infected_no=city.infected_no, medCity=city.medCity)
            combined_city_graph.add_node(city.cityName, population=city.population, infected_no=city.infected_no, medCity=city.medCity)
            for city_edge in city.neighbourCity:
                state_graphs[state_name].add_edge(city_edge.source.cityName, city_edge.destination.cityName, weight=city_edge.distance)
                combined_city_graph.add_edge(city_edge.source.cityName, city_edge.destination.cityName, weight=city_edge.distance)

                # Add edge label with distance
                state_graphs[state_name].edges[city_edge.source.cityName, city_edge.destination.cityName]['distance'] = city_edge.distance
                combined_city_graph.edges[city_edge.source.cityName, city_edge.destination.cityName]['distance'] = city_edge.distance

        # Add state interconnections
        for edge in state.neighbourState:
            state_interconnections.add_edge(edge.source, edge.destination, weight=edge.distance, direction=edge.direction)

    # Plot individual state graphs
    num_states = len(state_graphs)
    num_cols = math.ceil(math.sqrt(num_states))
    num_rows = math.ceil(num_states / num_cols)

    # Number of rows and columns for the subplot grid
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 8))
    fig.suptitle("Individual State Graphs", fontsize=16)

    # Iterate through each state graph and plot
    for i, (state_name, state_graph) in enumerate(state_graphs.items()):
        row = i // num_cols  # Calculate the row index for the subplot
        col = i % num_cols   # Calculate the column index for the subplot
        ax = axes[row, col]  # Get the subplot axis

       # Position nodes using the spring layout algorithm
        pos = nx.spring_layout(state_graph)
       # Draw the state graph
        nx.draw(state_graph, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8, font_weight="bold", ax=ax)
        edge_labels = nx.get_edge_attributes(state_graph, 'distance') # Get edge labels for distances
        nx.draw_networkx_edge_labels(state_graph, pos, edge_labels=edge_labels, ax=ax)
        ax.set_title(state_name)  # Set the title of the subplot

    # Remove empty subplots if needed
    for i in range(num_states, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])  

    # Adjust subplots to fit into the figure area
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Plot state interconnections
    plt.figure(figsize=(8, 6))
    pos_state_interconnections = nx.spring_layout(state_interconnections)
    nx.draw(state_interconnections, pos_state_interconnections, with_labels=True, node_size=3000, node_color="lightgreen", font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(state_interconnections, 'weight')
    nx.draw_networkx_edge_labels(state_interconnections, pos_state_interconnections, edge_labels=edge_labels, font_color='blue')
    plt.title("State Interconnections")
    plt.tight_layout()

    # Plot individual city graphs for each state
    for state_name, state_graph in state_graphs.items():
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(state_graph)
        nx.draw(state_graph, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8, font_weight="bold")
        edge_labels = nx.get_edge_attributes(state_graph, 'distance')
        nx.draw_networkx_edge_labels(state_graph, pos, edge_labels=edge_labels)
        plt.title(f"{state_name} City Graph")
        plt.tight_layout()

    # Plot combined city graph
    plt.figure(figsize=(12, 10))
    pos_combined = nx.spring_layout(combined_city_graph)
    nx.draw(combined_city_graph, pos_combined, with_labels=True, node_size=500, node_color="lightcoral", font_size=8, font_weight="bold")
    combined_edge_labels = nx.get_edge_attributes(combined_city_graph, 'distance')
    nx.draw_networkx_edge_labels(combined_city_graph, pos_combined, edge_labels=combined_edge_labels)
    plt.title("Combined City Graph")
    plt.tight_layout()

    plt.show() # Show all plots


def main():
    state_graph = StateGraph()
    city_graph = CityGraph()

    # Adding states
    # Adding states
    state_graph.addNewState("StateA", 5000000, 10000) #(statename,populatoin,infected)
    state_graph.addNewState("StateB", 3000000, 5000)
    state_graph.addNewState("StateC", 4000000, 7000)
    state_graph.addNewState("StateD", 3500000, 6000)
    state_graph.addNewState("StateE", 4500000, 8000)

    # Connecting states
    state_graph.addNeighbourState("StateA", "StateB", 100, "north")# connect states(add edge)
    state_graph.addNeighbourState("StateA", "StateC", 150, "east")
    state_graph.addNeighbourState("StateB", "StateD", 200, "west")
    state_graph.addNeighbourState("StateC", "StateE", 250, "south")
    state_graph.addNeighbourState("StateD", "StateE", 300, "northwest")
    
    # Adding cities
    city_graph.addCity("StateA", "CityA1", 100, 50, state_graph, True)  # Medical city
    city_graph.addCity("StateA", "CityA2", 200000, 2000, state_graph, True)
    city_graph.addCity("StateA", "CityA3", 150000, 1000, state_graph, True)
    city_graph.addCity("StateA", "CityA4", 100000, 500, state_graph)
    city_graph.addCity("StateA", "CityA5", 250000, 2500, state_graph, True)
    city_graph.addCity("StateA", "CityA6", 180000, 1500, state_graph)
    city_graph.addCity("StateA", "CityA7", 220000, 2000, state_graph)
    city_graph.addCity("StateA", "CityA8", 130000, 1000, state_graph)
    city_graph.addCity("StateA", "CityA9", 300000, 3500, state_graph, True)
    city_graph.addCity("StateB", "CityB1", 500000, 3000, state_graph)
    city_graph.addCity("StateB", "CityB2", 150000, 1500, state_graph)
    city_graph.addCity("StateB", "CityB3", 250000, 2000, state_graph)
    city_graph.addCity("StateB", "CityB4", 200000, 1800, state_graph)
    city_graph.addCity("StateB", "CityB5", 280000, 2200, state_graph)
    city_graph.addCity("StateB", "CityB6", 350000, 2500, state_graph, True)
    city_graph.addCity("StateB", "CityB7", 240000, 1800, state_graph)
    city_graph.addCity("StateB", "CityB8", 180000, 1500, state_graph, True)
    city_graph.addCity("StateB", "CityB9", 210000, 2000, state_graph)

    city_graph.addCity("StateC", "CityC1", 300000, 3500, state_graph, True)
    city_graph.addCity("StateC", "CityC2", 400000, 4000, state_graph)
    city_graph.addCity("StateC", "CityC3", 200000, 2500, state_graph, True)
    city_graph.addCity("StateC", "CityC4", 250000, 3000, state_graph)
    city_graph.addCity("StateC", "CityC5", 180000, 2000, state_graph)
    city_graph.addCity("StateC", "CityC6", 220000, 2300, state_graph, True)
    city_graph.addCity("StateC", "CityC7", 190000, 1800, state_graph)
    city_graph.addCity("StateC", "CityC8", 270000, 2800, state_graph, True)
    city_graph.addCity("StateC", "CityC9", 230000, 2100, state_graph)

    city_graph.addCity("StateD", "CityD1", 350000, 3500, state_graph)
    city_graph.addCity("StateD", "CityD2", 450000, 4500, state_graph)
    city_graph.addCity("StateD", "CityD3", 250000, 3000, state_graph, True)
    city_graph.addCity("StateD", "CityD4", 280000, 3200, state_graph)
    city_graph.addCity("StateD", "CityD5", 210000, 2500, state_graph, True)
    city_graph.addCity("StateD", "CityD6", 320000, 3800, state_graph)
    city_graph.addCity("StateD", "CityD7", 200000, 1800, state_graph, True)
    city_graph.addCity("StateD", "CityD8", 190000, 2000, state_graph)
    city_graph.addCity("StateD", "CityD9", 240000, 2300, state_graph)

    city_graph.addCity("StateE", "CityE1", 500000, 5000, state_graph, True)
    city_graph.addCity("StateE", "CityE2", 600000, 6000, state_graph)
    city_graph.addCity("StateE", "CityE3", 400000, 4500, state_graph)
    city_graph.addCity("StateE", "CityE4", 450000, 4800, state_graph)
    city_graph.addCity("StateE", "CityE5", 380000, 3800, state_graph)
    city_graph.addCity("StateE", "CityE6", 430000, 4200, state_graph)
    city_graph.addCity("StateE", "CityE7", 470000, 4900, state_graph, True)
    city_graph.addCity("StateE", "CityE8", 390000, 4000, state_graph)
    city_graph.addCity("StateE", "CityE9", 420000, 4300, state_graph, True) 


    # Connecting cities
    city_graph.connectCities("CityA1", "CityA3", 50)
    city_graph.connectCities("CityA1", "CityA2", 120)
    city_graph.connectCities("CityA1", "CityA5", 70)  # Connecting to the medical city
    city_graph.connectCities("CityA5", "CityA3", 10)
    city_graph.connectCities("CityA7", "CityA5", 20)
    city_graph.connectCities("CityA7", "CityA3", 30)
    city_graph.connectCities("CityA3", "CityA4", 40)
    city_graph.connectCities("CityA4", "CityA9", 50)
    city_graph.connectCities("CityA9", "CityA6", 60)
    city_graph.connectCities("CityA4", "CityA6", 70)
    city_graph.connectCities("CityA8", "CityA5", 80)
    city_graph.connectCities("CityA6", "CityA8", 90)

    city_graph.connectCities("CityB1", "CityB3", 50)
    city_graph.connectCities("CityB1", "CityB2", 120)
    city_graph.connectCities("CityB1", "CityB5", 70)  # Connecting to the medical city
    city_graph.connectCities("CityB5", "CityB3", 10)
    city_graph.connectCities("CityB7", "CityB5", 20)
    city_graph.connectCities("CityB7", "CityB3", 30)
    city_graph.connectCities("CityB3", "CityB4", 40)
    city_graph.connectCities("CityB4", "CityB9", 50)
    city_graph.connectCities("CityB9", "CityB6", 60)
    city_graph.connectCities("CityB4", "CityB6", 70)
    city_graph.connectCities("CityB8", "CityB5", 80)
    city_graph.connectCities("CityB6", "CityB8", 90)


    city_graph.connectCities("CityC1", "CityC3", 50)
    city_graph.connectCities("CityC1", "CityC2", 120)
    city_graph.connectCities("CityC1", "CityC5", 70)  # Connecting to the medical city
    city_graph.connectCities("CityC5", "CityC3", 10)
    city_graph.connectCities("CityC7", "CityC5", 20)
    city_graph.connectCities("CityC7", "CityC3", 30)
    city_graph.connectCities("CityC3", "CityC4", 40)
    city_graph.connectCities("CityC4", "CityC9", 50)
    city_graph.connectCities("CityC9", "CityC6", 60)
    city_graph.connectCities("CityC4", "CityC6", 70)
    city_graph.connectCities("CityC8", "CityC5", 80)
    city_graph.connectCities("CityC6", "CityC8", 90)


    city_graph.connectCities("CityD1", "CityD3", 50)
    city_graph.connectCities("CityD1", "CityD2", 120)
    city_graph.connectCities("CityD1", "CityD5", 70)  # Connecting to the medical city
    city_graph.connectCities("CityD5", "CityD3", 10)
    city_graph.connectCities("CityD7", "CityD5", 20)
    city_graph.connectCities("CityD7", "CityD3", 30)
    city_graph.connectCities("CityD3", "CityD4", 40)
    city_graph.connectCities("CityD4", "CityD9", 50)
    city_graph.connectCities("CityD9", "CityD6", 60)
    city_graph.connectCities("CityD4", "CityD6", 70)
    city_graph.connectCities("CityD8", "CityD5", 80)
    city_graph.connectCities("CityD6", "CityD8", 90)

    city_graph.connectCities("CityE1", "CityE3", 50)
    city_graph.connectCities("CityE1", "CityE2", 120)
    city_graph.connectCities("CityE1", "CityE5", 70)  # Connecting to the medical city
    city_graph.connectCities("CityE5", "CityE3", 10)
    city_graph.connectCities("CityE7", "CityE5", 20)
    city_graph.connectCities("CityE7", "CityE3", 30)
    city_graph.connectCities("CityE3", "CityE4", 40)
    city_graph.connectCities("CityE4", "CityE9", 50)
    city_graph.connectCities("CityE9", "CityE6", 60)
    city_graph.connectCities("CityE4", "CityE6", 70)
    city_graph.connectCities("CityE8", "CityE5", 80)
    city_graph.connectCities("CityE6", "CityE8", 90)

    city_graph.connectCities("CityA6", "CityB8", 90)
    city_graph.connectCities("CityB6", "CityC8", 90)
    city_graph.connectCities("CityC6", "CityD8", 90)
    city_graph.connectCities("CityD6", "CityE8", 90)

    # Reordering cities based on infection rate
    state_graph.reOrderCities() #call this funcition fo build th heap according to the infected in all states

    # Service the most infected city in a state
    state_graph.serviceCityInState("StateA") #extract max infected, find the shorest path
    state_graph.serviceCityInState("StateB")
    state_graph.serviceCityInState("StateC")
    state_graph.serviceCityInState("StateD")
    state_graph.serviceCityInState("StateE")
    
    state_graph.reOrderCities()  # call this function to build the heap according to the infected in all states
    # Processing commands
    inputs = int(input("Enter the number of commands: "))
    while inputs > 0:
        command = input("Enter a command: ")
        operation = command.split()

        if len(operation) == 0:
            print("No command entered.")
        elif operation[0] == "nearest_medical_city":
            city_name = operation[1] if len(operation) > 1 else "CityA1"
            nearest_medical_city = city_graph.getNearestMedicalCity(city_name)
            if nearest_medical_city:
                print("Nearest medical city:", nearest_medical_city.cityName)
            else:
                print("No nearest medical city found for the given city.")
        elif operation[0] == "infection_rate":
            state_name = operation[1] if len(operation) > 1 else "StateA"
            infection_rate = state_graph.calculateStateInfectionRate(state_name)
            if infection_rate is not None:
                print(f"Infection rate for {state_name}: {infection_rate}%")
            else:
                print(f"No data found for state: {state_name}")
        elif operation[0] == "find_bridges":
            bridges = city_graph.findBridges()
            print("Bridges in the city graph:", bridges)
        elif operation[0] == "find_articulation_points":
            articulation_points = city_graph.findArticulationPoints()
            print("Articulation points in the city graph:", articulation_points)
        elif operation[0] == "service_city":
            state_name = operation[1] if len(operation) > 1 else "StateA"
            state_graph.serviceCityInState(state_name)
        elif operation[0] == "get_max_state":
            max_state = state_graph.getmaxstate()
            print("State with maximum infection:", max_state)
        elif operation[0] == "generate_report":
            state_name = operation[1] if len(operation) > 1 else "StateA"
            state_graph.generatereport(state_name)
        elif operation[0] == "evacuate_city":
            city_name = operation[1] if len(operation) > 1 else "CityA2"
            city_graph.evacuatecity(city_name)
        elif operation[0] == "supply_state":
            quantity = int(operation[1]) if len(operation) > 1 else 4000
            state_name = operation[2] if len(operation) > 2 else "StateA"
            state_graph.supplyState(quantity, state_name)
        elif operation[0] == "reach_all_cities":
            state_name = operation[1] if len(operation) > 1 else "StateA"
            print("SHORTEST PATH TO REACH A CITY FROM MED-CITY:")
            print("\n")
            print(state_graph.reachAllCities(state_name))
            print("\n")
        elif operation[0] == "visualisation":
            visualisation(state_graph)
        else:
            print("Unknown command.")
        
        inputs -= 1


   
if __name__ == "__main__":
    main()
