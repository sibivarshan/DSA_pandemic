import heapq
from tabulate import tabulate

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
        self.density=infected/population


class cityNode:
    def __init__(self, state, C_name, population, infected,capacity=5000, med=False):
        self.stateName = state
        self.cityName = C_name
        self.population = population
        self.capacity=capacity
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
    
    def getmaxstate(self):
        max=self.stateVertices[0]
        for i in self.stateVertices:
            if max.density < i.density:
                max=i
        print("The State with the maximum affected people density is: ",max.StateName)

    def serviceCityInState(self, stateName):
        state = next((s for s in self.stateVertices if s.StateName == stateName), None)
        if state and state.cityList:
            mostInfectedCity = state.cityList[0]
            print("Serviced:", mostInfectedCity.cityName)
            self.moveToCityShortestPath(mostInfectedCity, state)
            state.cityList[0], state.cityList[-1] = state.cityList[-1], state.cityList[0]
            state.cityList.pop()
            self.maxHeapify(state.cityList, len(state.cityList), 0)
            state.mostInfected = state.cityList[0] if state.cityList else None

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


class CityGraph:
    def __init__(self):
        self.cityVertices = []

    def addCity(self, Sname, Cname, population, infected, stateGraph):
        temp_city = cityNode(Sname, Cname, population, infected)
        self.cityVertices.append(temp_city)
        stateGraph.addCitiesToStates(temp_city)

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
    state_graph.addNewState("StateA", 5000000, 10000)
    state_graph.addNewState("StateB", 3000000, 5000)

    # Connecting states
    state_graph.addNeighbourState("StateA", "StateB", 100, "north")

    # Adding cities
    city_graph.addCity("StateA", "CityA1", 100, 50, state_graph)
    city_graph.addCity("StateA", "CityA2", 200000, 2000, state_graph)
    city_graph.addCity("StateB", "CityB1", 500000, 3000, state_graph)
    city_graph.addCity("StateA", "CityA3", 150000, 1000, state_graph)  # Adding a medical city
    city_graph.cityVertices[0].medCity = True  # Setting CityA3 as a medical city
    city_graph.cityVertices[2].medCity = True
    # Connecting cities
    city_graph.connectCities("CityA1", "CityA2", 50)
    city_graph.connectCities("CityA1", "CityB1", 120)
    city_graph.connectCities("CityA1", "CityA3", 70)  # Connecting to the medical city
    city_graph.connectCities("CityA2", "CityA3", 10)

    # Reordering cities based on infection rate
    state_graph.reOrderCities()

    # Service the most infected city in a state
    state_graph.serviceCityInState("StateA")
    state_graph.serviceCityInState("StateB")
    state_graph.getmaxstate()
    state_graph.generatereport("StateA")


if __name__ == "__main__":
    main()
