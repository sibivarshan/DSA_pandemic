'''1.Create a separate branch to work with
   2.Do not change any variables and structure of this code
   3.Do not commit to main branch without complete checking
   4.Use valid names for the variables and functions
   5.Give code comments for all the functions,classes and all complex part of the code
   6.Make sure you dont modify others part without consulting.'''

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


class cityNode: #class city node is a class for the details of the city vertex
    def __init__(self, state, C_name, population, infected, med=False):
        self.stateName = state
        self.cityName = C_name
        self.population = population
        self.infected_no = infected
        self.distance = float('inf') #initially set to infinity
        self.parentCity = None
        self.medCity = med #only if its a medical city,this will be true
        self.neighbourCity = [] # edge list for city vertex


class edgeDist: 
#class to create edges in graphs of states and cities, it has source,destination and weight of the edge #direction can be ignored for now
    def __init__(self, sourceState, destinationState, distance, direction):
        self.source = sourceState
        self.destination = destinationState
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
            print("Serviced:", mostInfectedCity.cityName)
            self.moveToCityShortestPath(mostInfectedCity, state) # function to find the math to medical city
            state.cityList[0], state.cityList[-1] = state.cityList[-1], state.cityList[0]# first element is swapped with last
            state.cityList.pop() # remove that city which has been serviced
            self.maxHeapify(state.cityList, len(state.cityList), 0) #restore heap property
            state.mostInfected = state.cityList[0] if state.cityList else None


    '''This function is used to find the shortest path from the infected city to a medical city. 
    we first find the shortest path from the max node then travel in reverse hypothetically.
    used dijkstra's shortest path algorithm.'''
    def moveToCityShortestPath(self, city, stateVer):
        for ver in stateVer.cityList1: #initialize the distances
            ver.distance = float('inf') #infinity
            ver.parentCity = None
        city.distance = 0
        priority_queue = [] #using priority queue
        heapq.heappush(priority_queue, (city.distance, city)) # passing that queue as parameter for the heapq library, with city distance as key and city as value

        while priority_queue:
            current_distance, current_city = heapq.heappop(priority_queue) #similar to BFS, we do

            if current_city.medCity:
                print(f"Reached medical city: {current_city.cityName} with total distance {current_distance}")
                return current_distance # if a med city is reached, come out of the code

            for edge in current_city.neighbourCity:
                neighbor_city = edge.destination
                distance = edge.distance
                new_distance = current_distance + distance 

                if new_distance < neighbor_city.distance: # relaxation part, if v.d > u.d + w(u,v) then we change the ditance
                    neighbor_city.distance = new_distance
                    neighbor_city.parentCity = current_city
                    heapq.heappush(priority_queue, (new_distance, neighbor_city)) #change in queue since we may visit that vertex again 

        print("No medical city reachable")
        return float('inf')

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
    def addCity(self, Sname, Cname, population, infected, stateGraph):
        temp_city = cityNode(Sname, Cname, population, infected)
        self.cityVertices.append(temp_city)
        stateGraph.addCitiesToStates(temp_city)

    #add the edges to the city, edge list representation
    def connectCities(self, city1, city2, distance):
        tempCity1 = next((city for city in self.cityVertices if city.cityName == city1), None)
        tempCity2 = next((city for city in self.cityVertices if city.cityName == city2), None)

        if tempCity1 and tempCity2:
            tempCity1.neighbourCity.append(edgeDist(tempCity1, tempCity2, distance, None))
            tempCity2.neighbourCity.append(edgeDist(tempCity2, tempCity1, distance, None))


def main():
    state_graph = StateGraph()
    city_graph = CityGraph()

    # Adding states
    state_graph.addNewState("StateA", 5000000, 10000) #(statename,populatoin,infected)
    state_graph.addNewState("StateB", 3000000, 5000)

    # Connecting states
    state_graph.addNeighbourState("StateA", "StateB", 100, "north")# connect states(add edge)

    # Adding cities
    city_graph.addCity("StateA", "CityA1", 100, 50, state_graph) #add cities to respective graph and state
    city_graph.addCity("StateA", "CityA2", 200000, 2000, state_graph)
    city_graph.addCity("StateB", "CityB1", 500000, 3000, state_graph)
    city_graph.addCity("StateA", "CityA3", 150000, 1000, state_graph)  # Adding a medical city
    city_graph.cityVertices[0].medCity = True  # Setting CityA3 as a medical city
    city_graph.cityVertices[2].medCity = True
    # Connecting cities

    city_graph.connectCities("CityA1", "CityA2", 50)#connect cities in same state accordingly, connection to different state is also possible
    city_graph.connectCities("CityA1", "CityB1", 120)
    city_graph.connectCities("CityA1", "CityA3", 70)  # Connecting to the medical city
    city_graph.connectCities("CityA2", "CityA3", 10)

    # Reordering cities based on infection rate
    state_graph.reOrderCities() #call this funcition fo build th heap according to the infected in all states

    # Service the most infected city in a state
    state_graph.serviceCityInState("StateA") #extract max infected, find the shorest path
    state_graph.serviceCityInState("StateB")


if __name__ == "__main__":
    main()
