
# implement of A-PageRank Clustering for paper "A-PageRank Clustering: Spatial Structure Analysis Based on Space Syntax and Graph Analytics" appears in the 14th International Space Syntax Symposium


import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sknetwork.clustering import Louvain


# input：road network in shapefile
# output: road network with point ids, points extracted from road network


def preprocess(road):

    pl = []
    
    for index, row in road.iterrows():
    
    if len(row['geometry'].coords) == 2:
        
	p1 = Point(row['geometry'].coords[0][0], row['geometry'].coords[0][1])
        p2 = Point(row['geometry'].coords[-1][0],row['geometry'].coords[-1][1])
        
        if p1 not in pl:
            pl.append(p1)
            pfi = len(pl) - 1
	else:
            pfi = pl.index(p1)
        
        if p2 not in pl:
            pl.append(p2)
            psi = len(pl) - 1        
        else:
            psi = pl.index(p2)

        road.at[index, 'psf'] = str(int(pfi))
        road.at[index, 'pss'] = str(int(psi))    
        road.at[index, 'pef'] = str(int(psi))
        road.at[index, 'pes'] = str(int(pfi))            
        
        
    if len(row['geometry'].coords) > 2:
        
    	p1 = Point(row['geometry'].coords[0][0], row['geometry'].coords[0][1])
        p2 = Point(row['geometry'].coords[1][0], row['geometry'].coords[1][1])
        p3 = Point(row['geometry'].coords[-2][0],row['geometry'].coords[-2][1])    
        p4 = Point(row['geometry'].coords[-1][0],row['geometry'].coords[-1][1])    
    
        if p1 not in pl:
            pl.append(p1)
            psfi = len(pl) - 1
        else:
            psfi = pl.index(p1)
        
        if p2 not in pl:
            pl.append(p2)
            pssi = len(pl) - 1        
        else:
            pssi = pl.index(p2)    
    
        if p3 not in pl:
            pl.append(p3)
            pesi = len(pl) - 1
        else:
            pesi = pl.index(p3)
        
        if p4 not in pl:
            pl.append(p4)
            pefi = len(pl) - 1        
        else:
            pefi = pl.index(p4)        
    
        road.at[index, 'psf'] = str(int(psfi))
        road.at[index, 'pss'] = str(int(pssi))    
        road.at[index, 'pef'] = str(int(pefi))
        road.at[index, 'pes'] = str(int(pesi))

    df = pd.DataFrame()
    df['geometry'] = pl
    df['id'] = [x for x in range(len(df))]
    points = gpd.GeoDataFrame(df, geometry='geometry', crs='epsg:3857')
 
    return road, points


# input：preprocessed road network
# output: PageRank score matrix


def getPageRankScore(road):

    psf = sorted(list(set(road['psf'])))
    pef = sorted(list(set(road['pef'])))
    matrix = np.zeros((len(road), len(road)))
    matrix = pd.DataFrame(matrix, columns=[x for x in range(len(road))])
    matrix.index = [x for x in range(len(matrix))]
    pl = sorted(list(set(list(psf+ pef))))

    for p in pl:
        
    	s = road[(road['pef'] == p) | (road['psf'] == p)]
    	if len(s) > 1:
            inlist = list(s.index)
            for i in range(len(inlist)):
            	for j in range(i+1, len(inlist)):
                
                    interP = list(geo_df['geometry'])[int(p)]
                
                    if s.loc[inlist[i]]['pef'] == p:
                    	P1 = list(geo_df['geometry'])[int(s.loc[inlist[i]]['pes'])]
                    else:
                    	P1 = list(geo_df['geometry'])[int(s.loc[inlist[i]]['pss'])]                    
                    if s.loc[inlist[j]]['pef'] == p:
                    	P2 = list(geo_df['geometry'])[int(s.loc[inlist[j]]['pes'])]
                    else:
                    	P2 = list(geo_df['geometry'])[int(s.loc[inlist[j]]['pss'])]                    
                
                    angle =  (azimuth(interP, P1) - azimuth(interP, P2)).round(2)     
                    if angle >= 0:
                    	nangle = np.abs(180 - angle)
                    else:
                    	nangle = np.abs(180 + angle)
                                    
	            matrix.at[inlist[i], inlist[j]] = nangle
    	            matrix.at[inlist[j], inlist[i]] = nangle   	

    out = PageRank(matrix)
    return out


# input: road network, PageRank score matrix
# output: clusters of points in shapefile


def getClusters(road, points, matrix):

    louvain = Louvain()
    labels = louvain.fit_transform(matrix.values)

    psf = sorted(list(set(road['psf'])))
    pef = sorted(list(set(road['pef'])))
    pl = sorted(list(set(list(psf+ pef))))

    clusters = pd.DataFrame()
    clusters['id'] = pl
    clusters['label'] = label
    clusters = pd.merge(points, clusters, on=['id'], how='left')

    return clusters


# functions borrows from various resources
# angle calculation


def azimuth(point1, point2):
    '''azimuth between 2 shapely points (interval 0 - 360)'''
    angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
    return np.degrees(angle) if angle >= 0 else np.degrees(angle) + 360


# PageRank functions


def normalizeAdjacencyMatrix(A):
    n = len(A) 
    for j in range(len(A[0])):
        sumOfCol = 0
        for i in range(len(A)):
            sumOfCol += A[i][j]
        
        if sumOfCol == 0: 
            for val in range(n):
                A[val][j] = 1/n
        else:
            for val in range(n):
                A[val][j] = (A[val][j] / sumOfCol)
    return A

def dampingMatrix(A):
    n = len(A) 
    dampingFactor = 0.85
    Q = [[1/n]*n]*n
    arrA = np.array(A)
    arrQ = np.array(Q)
    arrM = np.add((dampingFactor)*arrA, (1-dampingFactor)*arrQ) # create damping matrix
    return arrM

def findSteadyState(M, n):
    evectors = np.linalg.eig(M)[1]
    
    eigenValues = np.linalg.eig(M)[0]
    lstEVals = []
    for val in eigenValues:
        lstEVals.append(round(val))
    
    idxWithEval1 = lstEVals.index(1)
    steadyStateVector = evectors[:, idxWithEval1]
    
    lstVersionSteadyState = []
    sumOfComps = 0
    returnVector = []
    for val in steadyStateVector:
        sumOfComps += val
        lstVersionSteadyState.append(val)
    for val in lstVersionSteadyState:
        returnVector.append(val/sumOfComps)
    return returnVector

def pageRank(A):
    n = len(A) 
    A = normalizeAdjacencyMatrix(A) 
    M = dampingMatrix(A) 
        
    steadyStateVectorOfA = findSteadyState(M, n)
    return steadyStateVectorOfA


