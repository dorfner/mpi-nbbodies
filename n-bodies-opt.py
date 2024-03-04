# Sequential version
# python3 n-bodies-base.py 12 1000

import sys
import math
import random
import matplotlib.pyplot as plt
import time
import numpy as np
from mpi4py import MPI

ID, POSX, POSY, SPEEDX, SPEEDY, WEIGHT = range(6)

solarmass=1.98892e30
def circlev(rx, ry):
    r2=math.sqrt(rx*rx+ry*ry)
    numerator=(6.67e-11)*1e6*solarmass
    return math.sqrt(numerator/r2)
# from http://physics.princeton.edu/~fpretori/Nbody/code.htm
def create_item(id, positionx, positiony, speedx, speedy, weight):
    if positionx == 0 and positiony == 0:  # the center of the world, very heavy one...
        speedx = 0
        speedy = 0
    else:
        if speedx==0 and speedy==0:            # initial values
            magv=circlev(positionx, positiony)
            absangle = math.atan(math.fabs(positiony/positionx))
            thetav= math.pi/2-absangle
            phiv = random.uniform(0,1)*math.pi
            speedx = -1*math.copysign(1, positiony)*math.cos(thetav)*magv
            speedy = math.copysign(1, positionx)*math.sin(thetav)*magv
            #Orient a random 2D circular orbit
            if (random.uniform(0,1) <=.5):
                speedx=-speedx
                speedy=-speedy
    return np.array([id, positionx, positiony, speedx, speedy, weight], dtype='f')

def str_item(item):
    return "ID="+str(item[ID])+" POS=("+str(item[POSX])+","+str(item[POSY])+") SPEED=("+str(item[SPEEDX])+","+str(item[SPEEDY])+") WEIGHT="+str(item[WEIGHT])

def display(m, l):
    for i in range(len(l)):
        print("PROC"+str(rank)+":"+m+"-"+str_item(l[i]))

def displayPlot(d):
    plt.gcf().clear()            # to remove to see the traces of the particules...
    plt.axis((-1e17,1e17,-1e17,1e17))
    xx = [ d[i][POSX]  for i in range(len(d)) ]
    yy = [ d[i][POSY]  for i in range(len(d)) ]
    plt.plot(xx, yy, 'ro')
    plt.draw()
    plt.pause(0.00001)            # in order to see something otherwise too fast...


def interaction(i, j):
    dist = math.sqrt( (j[POSX]-i[POSX])*(j[POSX]-i[POSX]) +  (j[POSY]-i[POSY])*(j[POSY]-i[POSY]) )
    if dist == 0:
        return np.zeros(2)
    g = 6.673e-11
    factor = g * i[WEIGHT] * j[WEIGHT] / (dist*dist+3e4*3e4)
    return np.array([factor*(j[POSX]-i[POSX])/dist, factor*(j[POSY]-i[POSY])/dist])

def update(d, f):
    dt = 1e11
    vx = d[SPEEDX] + dt * f[0]/d[WEIGHT]
    vy = d[SPEEDY] + dt * f[1]/d[WEIGHT]
    px = d[POSX] + dt * vx
    py = d[POSY] + dt * vy
    return create_item(d[ID], positionx=px, positiony=py, speedx=vx, speedy=vy, weight=d[WEIGHT])

def signature(world):
    s = 0
    for d in world:
        s+=d[POSX]+d[POSY]
    return s

def init_world(n):
    data = [ create_item(id=i, positionx=1e18*math.exp(-1.8)*(.5-random.uniform(0,1)), positiony=1e18*math.exp(-1.8)*(.5-random.uniform(0,1)), speedx=0, speedy=0, weight=(random.uniform(0,1)*solarmass*10+1e20)) for i in range(n-1)]
    data.append( create_item(id=nbbodies-1, positionx=0, positiony=0, speedx=0, speedy=0, weight=1e6*solarmass))
    return np.array(data)

nbbodies = int(sys.argv[1])
NBSTEPS = int(sys.argv[2])
DISPLAY = len(sys.argv) != 4

# Modify only starting here (and in the imports)

# To make a parallel version of the asymetric algorithm,
# we cannot simply give to each process nbbodies/size bodies to
# update, because that would be unoptimal :
#
#   Example with nbodies = 6 and size = 3
#     j\i012345 
#     0  .xxxxx
#     1  ..xxxx
#     2  ...xxx
#     3  ....xx
#     4  .....x
#        ^^^^^^
# rank:  001122
#
#   Here we can see that the process of rank 0 will only compute the forces of
# 1 body per step, while the process of rank 2 will compute the forces of
# 9 bodies ; this is not optimal.
#
# This is why we use this function:
#
#   Keeping the same example :
# OptimalBodiesDistribution(6,3) will return [3,4,5]
# This means 
# * ps of rank 0 will compute the forces of body 0 to 3
#       -> 6 iterations
# * ps of rank 1 will compute the forces of body 4
#       -> 4 iterations
# * ps of rank 2 will compute the forces of body 5
#       -> 5 iterations
#
# note: this function will add a small overhead (computed in O(n) (linear))

#  OptimalBodiesDistribution(N, size) returns a list of 'size'
# elements.
# Each element t[i] of the list is the index of the body of
# highest index process of rank i will need to update.
def optimalBodiesDistribution(N, size):
    # list to return
    indices = []
    # p is the number of iterations necessary in the asymetric algorithm divided by the number of processes
    p = (N*(N-1)) // (2*size)
    # seq is a sequence that helps compute the indices of the optimal distribution
    seq = 0

    for i in range(1,N):
        seq = (seq%p) + i
        if (seq >= p):
            indices += [i]

    return indices


# Init MPI values
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if (nbbodies % size != 0):
    print("The number of processes must be set to a multiple of the number of bodies")
    exit()

data = np.empty((nbbodies, 6), dtype='f')
indices = np.empty(size, dtype='i')
if (rank==0):
    random.seed(0)
    data = init_world(nbbodies)

    start_time = time.time()
    if DISPLAY:
        displayPlot(data)

    indices = np.array(optimalBodiesDistribution(nbbodies, size),
            dtype = 'i')
    
# each process will have its own share of the bodies to update
comm.Bcast(data, root=0)
comm.Bcast(indices, root=0)

lowerIndex = 0 if (rank==0) else indices[rank-1] +1
higherIndex = indices[rank] +1

# for the update we will need a simple data sharing like for n-bodies-par.py
shareSize = nbbodies // size
dataShare = np.empty((shareSize, 6), dtype='f')

for t in range(NBSTEPS):
    dataShare = data[rank*shareSize: (rank+1)*shareSize]

    force = np.zeros((nbbodies , 2), dtype='f8')
    forces_sum = np.zeros((nbbodies, 2), dtype='f8')

    for i in range(lowerIndex, higherIndex):
        for j in range(i):
            force_j_on_i = interaction(data[i], data[j])
            force[i] += force_j_on_i 
            force[j] -= force_j_on_i 
    comm.Allreduce(force, forces_sum, op=MPI.SUM)

    for i in range(shareSize):
        absoluteIndex = rank*shareSize + i
        dataShare[i] = update(dataShare[i], forces_sum[absoluteIndex])
    comm.Allgather(dataShare, data)

    if (rank==0 and DISPLAY):
        displayPlot(data)

if (rank==0):
    print("Duration : ", time.time()-start_time)
    print("Signature of the world :")
    print(signature(data))

