from collections import defaultdict
import math
import networkx as nx
import matplotlib.pyplot as plt
import heapq

class Statenode:
    def __init__(self, S_name, population, infected):
        self.StateName = S_name
        self.population = population
        self.infected_no = infected
        self.cityList = []
        self.cityList1 = []
        self.medCityList = []
        self.neighbourState = []
        self.mostInfected = None

class cityNode:
    def __init__(self, state, C_name, population, infected, med=False):
        self.stateName = state
        self.cityName = C_name
        self.population = population
        self.infected_no = infected
        self.distance = float('inf')
        self.parentCity = None
        self.medCity = med
        self.neighbourCity = []

class edgeDist:
    def __init__(self, sourceState, destinationState, distance, direction):
        self.source = sourceState
        self.destination = destinationState
        self.distance = distance
        self.direction = direction

class StateGraph:
    def __init__(self):
        self.stateVertices = []

    def addNewState(self, Sname, population, infected):
        newState = Statenode(Sname, population, infected)
        self.stateVertices.append(newState)

    def addNeighbourState(self, state1, state2, dist, direction):
        state1_vertex = next((state for state in self.stateVertices if state.StateName == state1), None)
        state2_vertex = next((state for state in self.stateVertices if state.StateName == state2), None)

        if state1_vertex and state2_vertex:
            state1_vertex.neighbourState.append(edgeDist(state1, state2, dist, direction))
            state2_vertex.neighbourState.append(edgeDist(state2, state1, dist, direction))

    def addCitiesToStates(self, cityVertex):
        for stateVertex in self.stateVertices:
            if stateVertex.StateName == cityVertex.stateName:
                stateVertex.cityList.append(cityVertex)
                stateVertex.cityList1.append(cityVertex)
                if cityVertex.medCity:
                    stateVertex.medCityList.append(cityVertex)

    def reOrderCities(self):
        for stateVertex in self.stateVertices:
            stateVertex.cityList = self.buildCityHeap(stateVertex.cityList, len(stateVertex.cityList))
            stateVertex.mostInfected = stateVertex.cityList[0] if stateVertex.cityList else None

    def maxHeapify(self, cityList, N, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < N and cityList[l].infected_no > cityList[largest].infected_no:
            largest = l
        if r < N and cityList[r].infected_no > cityList[largest].infected_no:
            largest = r
        if largest != i:
            cityList[i], cityList[largest] = cityList[largest], cityList[i]
            self.maxHeapify(cityList, N, largest)

    def buildCityHeap(self, cityList, N):
        semi_root = N // 2
        for i in range(semi_root - 1, -1, -1):
            self.maxHeapify(cityList, N, i)
        return cityList

    def serviceCityInState(self, stateName):
        state = next((s for s in self.stateVertices if s.StateName == stateName), None)
        if state and state.cityList:
            mostInfectedCity = state.cityList[0]
            print("Serviced:", mostInfectedCity.cityName)
            self.moveToCityShortestPath(mostInfectedCity, state)
            state.cityList[0], state.cityList[-1] = state.cityList[-1], state.cityList[0]
            self.maxHeapify(state.cityList, len(state.cityList), 0)
            state.mostInfected = state.cityList[0] if state.cityList else None

    def calculateStateInfectionRate(self, stateName):
        state = next((s for s in self.stateVertices if s.StateName == stateName), None)
        if state:
            return (state.infected_no / state.population) * 100
        return None

    def moveToCityShortestPath(self, city, stateVer):
        for ver in stateVer.cityList1:
            ver.distance = float('inf')
            ver.parentCity = None
        city.distance = 0
        priority_queue = []
        heapq.heappush(priority_queue, (city.distance, city))

        while priority_queue:
            current_distance, current_city = heapq.heappop(priority_queue)
            if current_city.medCity:
                print(f"Reached medical city: {current_city.cityName} with total distance {current_distance}")
                return current_distance

            for edge in current_city.neighbourCity:
                neighbor_city = edge.destination
                distance = edge.distance
                new_distance = current_distance + distance

                if new_distance < neighbor_city.distance:
                    neighbor_city.distance = new_distance
                    neighbor_city.parentCity = current_city
                    heapq.heappush(priority_queue, (new_distance, neighbor_city))

        print("No medical city reachable")
        return float('inf')

    def prim(self, state_name):
        state = next((s for s in self.stateVertices if s.StateName == state_name), None)
        if not state:
            return []

        start_node = state.medCityList[0]
        graph = defaultdict(dict)

        # Select only the cities within the specified state
        state_cities = {city.cityName: city for city in state.cityList}

        for city in state.cityList:
            for neighbor in city.neighbourCity:
                if neighbor.destination.cityName in state_cities:
                    graph[city.cityName][neighbor.destination.cityName] = neighbor.distance

        mst = []
        visited = set([start_node.cityName])

        while len(visited) < len(state_cities):
            min_edge = None
            min_weight = float('inf')
            for source_city in visited:
                for dest_city, weight in graph[source_city].items():
                    if dest_city not in visited and weight < min_weight:
                        min_edge = (source_city, dest_city)
                        min_weight = weight

            if min_edge:
                source, dest = min_edge
                mst.append((source, dest, graph[source][dest]))
                visited.add(dest)
        for city in state.cityList1:
            print(city.cityName)
        return mst



class CityGraph:
    def __init__(self):
        self.cityVertices = []

    def addCity(self, Sname, Cname, population, infected, stateGraph,med=False):
        temp_city = cityNode(Sname, Cname, population, infected,med)
        self.cityVertices.append(temp_city)
        stateGraph.addCitiesToStates(temp_city)

    def connectCities(self, city1, city2, distance):
        tempCity1 = next((city for city in self.cityVertices if city.cityName == city1), None)
        tempCity2 = next((city for city in self.cityVertices if city.cityName == city2), None)

        if tempCity1 and tempCity2:
            tempCity1.neighbourCity.append(edgeDist(tempCity1, tempCity2, distance, None))
            tempCity2.neighbourCity.append(edgeDist(tempCity2, tempCity1, distance, None))

    def getNearestMedicalCity(self, cityName):
        city = next((c for c in self.cityVertices if c.cityName == cityName), None)
        if city:
            for c in self.cityVertices:
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

def visualisation(state_graph):
    state_interconnections = nx.DiGraph()
    state_graphs = {}
    combined_city_graph = nx.DiGraph()

    for state in state_graph.stateVertices:
        state_name = state.StateName
        state_graphs[state_name] = nx.DiGraph()

        for city in state.cityList:
            state_graphs[state_name].add_node(city.cityName, population=city.population, infected_no=city.infected_no, medCity=city.medCity)
            combined_city_graph.add_node(city.cityName, population=city.population, infected_no=city.infected_no, medCity=city.medCity)
            for city_edge in city.neighbourCity:
                state_graphs[state_name].add_edge(city_edge.source.cityName, city_edge.destination.cityName, weight=city_edge.distance)
                combined_city_graph.add_edge(city_edge.source.cityName, city_edge.destination.cityName, weight=city_edge.distance)
                state_graphs[state_name].edges[city_edge.source.cityName, city_edge.destination.cityName]['distance'] = city_edge.distance
                combined_city_graph.edges[city_edge.source.cityName, city_edge.destination.cityName]['distance'] = city_edge.distance

        for edge in state.neighbourState:
            state_interconnections.add_edge(edge.source, edge.destination, weight=edge.distance, direction=edge.direction)

    num_states = len(state_graphs)
    num_cols = math.ceil(math.sqrt(num_states))
    num_rows = math.ceil(num_states / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    for idx, (state_name, state_g) in enumerate(state_graphs.items()):
        row = idx // num_cols
        col = idx % num_cols
        pos = nx.spring_layout(state_g)
        nx.draw(state_g, pos, with_labels=True, node_size=700, ax=axs[row, col])
        edge_labels = nx.get_edge_attributes(state_g, 'distance')
        nx.draw_networkx_edge_labels(state_g, pos, edge_labels=edge_labels, ax=axs[row, col])
        axs[row, col].set_title(state_name)

    combined_pos = nx.spring_layout(combined_city_graph)
    plt.figure()
    nx.draw(combined_city_graph, combined_pos, with_labels=True, node_size=700)
    combined_edge_labels = nx.get_edge_attributes(combined_city_graph, 'distance')
    nx.draw_networkx_edge_labels(combined_city_graph, combined_pos, edge_labels=combined_edge_labels)

    state_pos = nx.spring_layout(state_interconnections)
    plt.figure()
    nx.draw(state_interconnections, state_pos, with_labels=True, node_size=700)
    state_edge_labels = nx.get_edge_attributes(state_interconnections, 'weight')
    nx.draw_networkx_edge_labels(state_interconnections, state_pos, edge_labels=state_edge_labels)

    plt.show()

def main():
    state_graph = StateGraph()
    city_graph = CityGraph()

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
    city_graph.addCity("StateA", "CityA1", 100, 50, state_graph) #add cities to respective graph and state
    city_graph.addCity("StateA", "CityA2", 200000, 2000, state_graph)
    city_graph.addCity("StateA", "CityA3", 150000, 1000, state_graph,True)  # Adding a medical city
    city_graph.addCity("StateA", "CityA4", 100000, 500, state_graph)
    city_graph.addCity("StateA", "CityA5", 250000, 2500, state_graph)

    city_graph.addCity("StateB", "CityB1", 500000, 3000, state_graph)
    city_graph.addCity("StateB", "CityB2", 150000, 1500, state_graph)
    city_graph.addCity("StateB", "CityB3", 250000, 2000, state_graph)

    city_graph.addCity("StateC", "CityC1", 300000, 3500, state_graph)
    city_graph.addCity("StateC", "CityC2", 400000, 4000, state_graph)
    city_graph.addCity("StateC", "CityC3", 200000, 2500, state_graph)

    city_graph.addCity("StateD", "CityD1", 350000, 3500, state_graph)
    city_graph.addCity("StateD", "CityD2", 450000, 4500, state_graph)
    city_graph.addCity("StateD", "CityD3", 250000, 3000, state_graph)

    city_graph.addCity("StateE", "CityE1", 500000, 5000, state_graph)
    city_graph.addCity("StateE", "CityE2", 600000, 6000, state_graph)
    city_graph.addCity("StateE", "CityE3", 400000, 4500, state_graph) 
    
    
    # Adding a medical city
    city_graph.cityVertices[2].medCity = True  # Setting CityA3 as a medical city
    city_graph.cityVertices[5].medCity = True  # Setting CityB1 as a medical city
    city_graph.cityVertices[8].medCity = True  # Setting CityC1 as a medical city
    city_graph.cityVertices[11].medCity = True  # Setting CityD1 as a medical city
    city_graph.cityVertices[14].medCity = True  # Setting CityE1 as a medical city
    
    
    # Connecting cities
    city_graph.connectCities("CityA1", "CityA2", 50)#connect cities in same state accordingly, connection to different state is also possible
    city_graph.connectCities("CityA1", "CityA3", 70)  # Connecting to the medical city
    city_graph.connectCities("CityA2", "CityA3", 10)
    city_graph.connectCities("CityA3", "CityA4", 20)
    city_graph.connectCities("CityA4", "CityA5", 30)

    city_graph.connectCities("CityB1", "CityB2", 40)
    city_graph.connectCities("CityB2", "CityB3", 50)

    city_graph.connectCities("CityC1", "CityC2", 60)
    city_graph.connectCities("CityC2", "CityC3", 70)

    city_graph.connectCities("CityD1", "CityD2", 80)
    city_graph.connectCities("CityD2", "CityD3", 90)

    city_graph.connectCities("CityE1", "CityE2", 100)
    city_graph.connectCities("CityE2", "CityE3", 110)

    # Reordering cities based on infection rate
    state_graph.reOrderCities() #call this funcition fo build th heap according to the infected in all states
    
    # Service the most infected city in a state
    state_graph.serviceCityInState("StateA") #extract max infected, find the shorest path
    state_graph.serviceCityInState("StateB")
    state_graph.serviceCityInState("StateC")
    state_graph.serviceCityInState("StateD")
    state_graph.serviceCityInState("StateE")
    print("SHORTEST PATH TO REACH A CITY FROM MED-CITY:")
    print("\n")
    print(state_graph.prim("StateA"))
    print("\n")
    visualisation(state_graph)
    
    nearest_medical_city = city_graph.getNearestMedicalCity("CityA1")  

    # Print the result
    if nearest_medical_city:
        print("Nearest medical city:", nearest_medical_city.cityName)
    else:
        print("No nearest medical city found for the given city.")


    state_name = "StateA"  # Replace with the desired state name
    infection_rate = state_graph.calculateStateInfectionRate(state_name)
    if infection_rate is not None:
        print(f"Infection rate for {state_name}: {infection_rate}%")
    else:
        print(f"No data found for state: {state_name}")
    

            
if __name__ == "__main__":
    main()
