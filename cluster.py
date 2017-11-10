import heapq
import math
from pyspark import SparkContext
from pyspark import SparkConf
import sys



iris_plant=sys.argv[1]
k_number=str(sys.argv[2])
sc = SparkContext("local[1]", "Simple App")
data=sc.textFile(iris_plant)
#rate=ratingsData.map(lambda x:x.split(",")).map(lambda x:(x[0],x[1],x[2]))
plantData=data.map(lambda x:x.split(",")).map(lambda x:(x[0],x[1],x[2],x[3],x[4]))
Data=plantData.map(lambda x:(float(x[0]),float(x[1]),float(x[2]),float(x[3]),str(x[4]))).collect()

def specificData(data):
    i=0
    wholelist=[]
    while i<len(data):
        coordinatedata=data[i]
        indexcoordinatelist=[]
        j=0
        indexlist=[]
        indexlist.append(i+1)
        indexcoordinatelist.append(indexlist)
        while j<len(coordinatedata)-1:
            item=coordinatedata[j]
            indexcoordinatelist.append(item)
            j=j+1
        wholelist.append(indexcoordinatelist)
        i=i+1
    return wholelist
cleanData=specificData(Data)


def getDistance(cleandata):
    distancelist=[]
    for i in range(len(cleandata)-1):
        aindex=cleandata[i][0]
        x1=cleandata[i][1]
        x2 = cleandata[i][2]
        x3 = cleandata[i][3]
        x4 = cleandata[i][4]
        for j in range(i + 1, len(cleandata), 1):
            bindex=cleandata[j][0]
            y1 = cleandata[j][1]
            y2 = cleandata[j][2]
            y3 = cleandata[j][3]
            y4 = cleandata[j][4]
            distance = math.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2 + (x3 - y3) ** 2 + (x4 - y4) ** 2)
            comboindex=aindex+bindex
            heapq.heappush(distancelist, [distance,comboindex])
    distancelist=sorted(distancelist)
    return distancelist



def getShortest(distance):
    shortdistance=heapq.heappop(distance)
    return shortdistance
#shortestDistance=getShortest(distanceHeap) #[0.0,[10,35]]

#pair=shortestDistance[1]
#print(pair)  #[10,35]


def getCentroidCoor(cluster):
    i = 0
    x = 0
    y = 0
    z = 0
    w = 0
    centroidlist = []
    indexlist = []
    while i < len(cluster):
        index = cluster[i]
        indexlist.append(index)
        x = x + cleanData[index - 1][1]
        y = y + cleanData[index - 1][2]
        z = z + cleanData[index - 1][3]
        w = w + cleanData[index - 1][4]
        i = i + 1
    centroidlist.append(indexlist)
    x, y, z, w = x / len(cluster), y / len(cluster), z / len(cluster), w / len(cluster)
    centroidlist.append(x)
    centroidlist.append(y)
    centroidlist.append(z)
    centroidlist.append(w)
    return centroidlist
#newCentroid=getCentroidCoor(pair)
#print(newCentroid)        #[[10,35],4.9,3.1,1.5,0.1]

def filtercoordinate(newcentroid,updatedata):
    filtered_coordinate = list(filter(lambda x: len(set(x[0]).intersection(set(newcentroid[0]))) == 0, updatedata))
    filtered_coordinate.append(newcentroid)
    return filtered_coordinate
#newdata=filtercoordinate(newCentroid,updataData)
#print(newdata)


#newdatadistance=getDistance(newdata)
#print(newdatadistance)

def preliminaryResult():
    k=int(k_number)
    mergedtime=150-k
    i=0
    cleanData=specificData(Data)
    updataData=cleanData
    while i<mergedtime:
        distanceHeap=getDistance(updataData)
        shortestDistance = getShortest(distanceHeap)  # [0.0,[10,35]]
        pair = shortestDistance[1]                    #[10,35]
        newCentroid = getCentroidCoor(pair)           #[[10,35],4.9,3.1,1.5,0.1]
        updataData =filtercoordinate(newCentroid, updataData)
        i=i+1
    return updataData
clusterResult=preliminaryResult()


def ClusterList(clusterresult):
    i = 0
    clusterlist = []
    while i < len(clusterresult):
        clusterlist.append(clusterresult[i][0])
        i = i + 1
    return clusterlist
clusterList=ClusterList(clusterResult) #[[61, 99, 58, 94], [107, 63, 67, 85, 62, 56, 91, 89, 96, 97, 95, 100, 68, 83, 93, 6





def clustercoordinatelist(clusterlist):
    wholelist=[]
    for cluster in clusterlist:
        coordinatelist=[]
        for index in cluster:
            coordinate=Data[index-1]
            coordinatelist.append(coordinate)
        wholelist.append(coordinatelist)
    return wholelist
clusterCoordinateList=clustercoordinatelist(clusterList)


def output(clusterlist):
    clustername=[]
    for cluster in clusterList:
        setosa_number=0
        versicolor_number=0
        virginica_number=0
        for index in cluster:
            if Data[index-1][4]=="Iris-setosa":
                setosa_number=setosa_number+1
            elif Data[index-1][4]=="Iris-versicolor":
                versicolor_number=versicolor_number+1
            else:
                virginica_number =virginica_number+1
        if setosa_number==max(setosa_number,versicolor_number,virginica_number):
            clustername.append("Iris-setosa")
        elif versicolor_number==max(setosa_number,versicolor_number,virginica_number):
            clustername.append("Iris-versicolor")
        else:
            clustername.append("Iris-virginica")
    return clustername
clusterName=output(clusterList)

def wrongClassification(clusterList,clusterName):
    number=0
    i=0
    while i<len(clusterName):
        plantname=clusterName[i]
        cluster=clusterList[i]
        j=0
        while j<len(cluster):
            index=cluster[j]
            if Data[index-1][4]!=plantname:
                number=number+1
            j=j+1
        i=i+1
    return number
wrong_number=wrongClassification(clusterList,clusterName)

def write_output():
    file = open("/Users/yuxianghou/Desktop/Yuxiang_Hou_Cluster.txt", "w")
    i=0
    while i<len(clusterList):
        clusterCoordinate=clusterCoordinateList[i]
        clusterCoordinate1 = map(lambda x: "[" + str(x[0]) + "," + str(x[1]) + "," + str(x[2]) + "," + str(x[3]) + "," + str(x[4]) + "]",clusterCoordinate)
        file.write("cluster:"+clusterName[i]+"\n")
        for coordinate in clusterCoordinate1:
            file.write(coordinate)
            file.write("\n")
        file.write("Number of points in this cluster:"+str(len(clusterCoordinate)))
        file.write("\n")
        file.write("\n")
        i=i+1
    file.write("\n")
    file.write("Number of points wrongly assigned:"+str(wrong_number))
    file.close()
write_output()

