
#class to represent all the states using only the statenode type
class Statenode:

    def __init__(self,S_name,population,infected):
        self.StateName = S_name
        self.population = population
        self.infected_no = infected
        self.cityList = []
        self.medCityList = []
        self.neighbourState = []
        self.mostInfected = None

#class to represent all the city nodes as citynode type
class cityNode:

    def __init__(self,state,C_name,population,infected,med=True):
        self.stateName = state
        self.cityName = C_name
        self.population = population
        self.infected_no = infected
        self.leftCityI = None
        self.rightCityI = None
        self.medCity = med
        self.neighbourCity = []

#class creates edge object that contains the distance from one vertex to another and direcction
class edgeDist:

    def __init__(self,sourceState,destinationState,distance,dir):
        self.source = sourceState
        self.destination = destinationState
        self.distance = distance
        self.direction = dir
    
#the state class to have all the state nodes 
class state:

    def __init__(self):
        self.stateVertices = [] 

    def addNewState(self,Sname,population,infected):
        newState = Statenode(Sname,population,infected)
        self.stateVertices.append(newState)

    def addNeighbourState(self,state1,state2,dist): #direction from first state(north,south,east,west)

        #if both states are there to create an edge
        if (state1 and state2 in self.stateVertexs):
            #create an edge object
            state1.neigbourState.append(edgeDist(state1,state2,dist))  #append the states to each others adjecency list
            state2.neigbourState.append(edgeDist(state2,state1,dist))

    #this function adds the newly entered city to the respective state.
    def addCitiesToStates(self,cityVertex):
        for stateVertex in self.stateVertices:
            if stateVertex.StateName == cityVertex.StateName:
                stateVertex.cityList.append(cityVertex)
                if cityVertex.medCity == True:
                    stateVertex.medCityList.append(cityVertex)
        
            
    #once the cities are added , they are put into the priority queue to obbtain the most infected city for service providance
    def reOrderCities(self):

        for stateVertex in self.stateVertices:
            stateVertex.cityList = self.buildCityHeap(stateVertex.cityList)
            stateVertex.mostInfected = stateVertex.cityList[0]

    #max heapify function to reorder based on the infected list in the priority queue
    def maxHeapify(self,city, N, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < N and city[l].infected_no > city[largest].infected_no:
            largest = l
        if r < N and city[r].infected_no > city[largest].infected_no:
            largest = r
        if largest != i:
            city[i].infected_no, city[largest].infected= city[largest].infected_no, city[i].infected_no
            self.maxHeapify(city, N, largest)

    #builds a max heap
    def buildCityHeap(self,cityList,N):
        semi_root = N // 2
        for i in range(semi_root - 1, -1, -1):
            self.maxHeapify(cityList, N, i)
        return cityList
    

    #given a state, the medicare is servicing the most infected city in teh state , by extracting the max infected from cityList
    def servieCityInSate(self,state):
        if self.size == 0:
            return None
        print("serviced:", state.mostInfected)
        state.cityList[0], state.cityList[self.size - 1] = state.cityList[self.size -1], state.cityList[0]
        state.cityList.pop()
        self.size -= 1
        self.maxHeapify(state.cityList, self.size, 0)
        state.mostInfected = state.cityList[0]

            

#class to represent all the city vertices in the graph
class city:

    def __init__(self):
        self.cityVertices = []

    #function to add a new city to the map
    def addCity(self,Sname,Cname,population,infected):
        temp_city = cityNode(Sname,Cname,population,infected)
        self.cityVertices.append(temp_city)
        state.addCitiesToStates(temp_city)# calls the funciton to put this city in its respective state list


    #connects the cities 
    def connectCities(self,city1,city2,distance):
        tempCity1 = None
        tempCity2 = None
        for cName in self.cityVertices: #gets the vertices of 2 cities
            if cName.cityName == city1:
                tempCity1 = cName
            if cName.cityName == city2:
                tempCity2 = cName

        if tempCity2 and tempCity1: #true if both the cities are found
            tempCity1.neighbourCity.append(edgeDist(city1,city2,distance)) #append the cityes to each others adjecency list
            tempCity2.neighbourCity.append(edgeDist(city2,city1,distance))
               


    def moveToCityShortestPath(self):
        pass
        


    
    
    