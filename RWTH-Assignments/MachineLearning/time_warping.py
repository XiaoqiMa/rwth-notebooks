#/usr/bin/python3

# Statistical Classification and Machine Learning WS 17/18
# Exercise 12, Task 1

# If you need a 'inf', you can use float("inf")
# which behaves as a infinite number w.r.t. addition.

# Please make use of the functions and the variables
# provided in this block when you write your answer codes.

# You can search for "TODO" to find your tasks.

import argparse
import numpy as np
import math

# Argument options
parser = argparse.ArgumentParser(description='Statistical Classification and Machine Learning Exercise 12 Task 1')
parser.add_argument('implementation', help='recursive|memoization|iterative')
parser.add_argument('data', help='full|small')
args = parser.parse_args()

# Utilities
def Read(filename):
    data = []
    # Read function
    with open(filename) as f:
        data_str = f.readlines()
        for i in range(0, len(data_str)):
            data.append(float(data_str[i].rstrip()))
        return data

def Init():
    # Initialize the computation counter by 0.
    global counter
    counter = 0
    # Initialize the table of size TxS by -1.
    global D
    D = -np.ones([T,S])
    print("Initialization done.")

def dist(x,y):  # l1 distance.
    global counter
    # We count the number of distance computation.
    counter = counter + 1
    return np.abs(x-y)

# Introduction: Read the data
# This is another helper code block which introduces
# the name of variables to be used in your solution.

# The tasks start at the next block.

# Download the data from L2P and put the paths here:
if args.data == 'small':
    # Small data for quick test:
    x_data = "./small_x.dat"
    r_data = "./small_r.dat"
elif args.data == 'full':
    # Full data:
    x_data = "./x.dat"
    r_data = "./r.dat"
else:
    print("'data' should be one of the two: small|full")

x = Read(x_data)
r = Read(r_data)

# Please use the variable names 'x' and 'r' in your
# answer code.

# Define some global variables
T = len(x)
S = len(r)
maxJump = 2
jumpPen = [2, 0, 2] # time distortion penalties

# Define containers
D = -np.ones([T,S])
counter = 0

# ============================================================
#   Task 1 (b)
# ============================================================

def Recursive():
    print("Task 1 (b) ====================================")
    # TODO: Your task is to implement a helper
    # function "_recursive" below
    return _recursive(T-1,S-1)

def _recursive(t,s):
     # reached end of allowed path
    if t == 0 and s == 0:
        return dist(x[t], r[s])
    # invalid path
    elif (t == 0 and s != 0) or (t != 0 and s == 0):
        return float('inf')
    # otherwise we return the recursion algo with time distortion
    return dist(x[t], r[s]) + min([_recursive(t-1, s) + jumpPen[0], _recursive(t-1, s-1) + jumpPen[1], _recursive(t-1, s-2) + jumpPen[2]])

# ============================================================
#   Task 1 (c)
# ============================================================

def Memoization():
    print("Task 1 (c) ====================================")
    # TODO: Your task is to implement a helper
    # function "_memoize" below
    return _memoize(T-1, S-1)


def _memoize(t, s):
    if D[t][s] < 0:
        # analogous to above
        if t == 0 and s == 0:
            d = dist(x[t], r[s])
        elif (t == 0 and s != 0) or (t != 0 and s == 0):
            d = float('inf')
        else:
            d = dist(x[t], r[s]) + min([_memoize(t-1, s) + jumpPen[0], _memoize(t-1, s-1) + jumpPen[1], _memoize(t-1, s-2) + jumpPen[2]])
        D[t][s] = d
    return D[t][s]

# ============================================================
#   Task 1 (d)
# ============================================================

def Iterative():
    print("Task 1 (d) ===========================")
    return _iterative()

def _iterative():
    D[0] = np.array([dist(x[0], r[0])] + [float('inf') for s in range(S - 1)])
    for t in range(1, T):
        for s in range(S):
            if s > 1:
                D[t][s] = dist(x[t], r[s]) + min(D[t - 1][s] + jumpPen[0], D[t - 1][s - 1] + jumpPen[1], D[t - 1][s - 2] + jumpPen[2])
            # for lower values of s we cant use all transitions
            # without escaping the grid, so limit them accordingly
            elif s == 1:
                D[t][s] = dist(x[t], r[s]) + min(D[t - 1][s] + jumpPen[0], D[t - 1][s - 1] + jumpPen[1])
            elif s == 0:
                D[t][s] = dist(x[t], r[s]) + D[t - 1][s] + jumpPen[0]
    return D[T - 1][S - 1]

# Main computation starts here.
Init()
if args.implementation == 'recursive':
    if args.data == 'full':
        print("Recursive implementation on the full data will run forever. Exiting...")
    else:
        print("[Small data] Global distance is", Recursive())
        print("    which should be 3101.0")
        print("[Small data] Number of computation is", counter)
        print("    which should be 19238")
elif args.implementation == 'memoization':
    if args.data == 'full':
        print("Global distance we get is", Memoization())
        print("    which should be 14925.0")
        print("Number of computation of our code is", counter)
        print("    which should be 5407")
    else:
        print("[Small data] Global distance is", Memoization())
        print("    which should be 3101.0")
        print("[Small data] Number of computation of our code is", counter)
        print("    which should be 75")
elif args.implementation == 'iterative':
    if args.data == 'full':
        print("Global distance is", Iterative())
        print("    which should be 14925.0")
        print("Number of computation is", counter)
        print("    which should be equal to 6967 or better.")
    else:
        print("[Small data] Global distance is", Iterative())
        print("    which should be 3101.0")
        print("[Small data] Number of computation is", counter)
        print("    which should be equal to 91 or better.")
else:
    print("'implementation' should be one of the three: recursive|memoization|iterative")
