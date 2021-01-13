import numpy as np
from numpy.linalg import inv
from numpy.linalg import eig
import matplotlib.pyplot as plt
import time
'''
README: Run the file, uncomment lines 21, 25, 29(the lines running extended_kalman_filter code)
one after the other to see the plots, error values and coordinates. Type main() and run.
Change lambda at your own peril!
'''
def main():
  beacon_positions = read_position_from_csv("beacons.csv")

  # TODO find a random point inside a polygon as the initial position
  receiver_position = np.array([-3, 10])
  lamda = 1.3

  stationary_distances = read_distance_from_csv("stationary_z.csv")
  ## Following are three scenarios for the extended kalman filter.
  max_iterations = stationary_distances.shape[0]

  # Stationary receiver - stationary model
  extended_kalman_filter(beacon_positions, receiver_position, stationary_distances, max_iterations, lamda, "static")

  # Moving receiver - stationary model
  moving_distances = read_distance_from_csv("moving_z.csv")
  #extended_kalman_filter(beacon_positions, receiver_position, moving_distances, max_iterations, lamda, "static")

  # moving receiver - moving model
  m_state_vector = np.array([-3, 10, 0, 0]) #x, y, ux, uy
  #extended_kalman_filter(beacon_positions, m_state_vector, moving_distances, max_iterations, lamda, "dynamic")
  pass

def calculate_position(m_state_vector, t):
  position = []
  position[0] = m_state_vector[0] + t*(m_state_vector[2])
  position[1] = m_state_vector[1] + t*(m_state_vector[3])
  return position

def calculate_distance(bp_x, bp_y, receiver_position):
  distance = np.sqrt(((receiver_position[0] - bp_x)**2) + ((receiver_position[1] - bp_y)**2))
  #print("distance", distance)
  return distance

def compute_g(beacon_positions, receiver_position, actual_distances):
  g = np.zeros((beacon_positions.shape[1]))
  #print("g-shape",g.shape[0])
  for index in range(g.shape[0]):
    g[index] = actual_distances[index] - calculate_distance(beacon_positions[0][index], beacon_positions[1][index], receiver_position)
    #print("g",index, g[index])
  return g

def compute_dg(beacon_positions, receiver_position):
  dg = np.zeros(beacon_positions.shape)
  for index in range(beacon_positions.shape[1]):
    t = calculate_distance(beacon_positions[0][index], beacon_positions[1][index], receiver_position)
    #print(beacon_positions)
    dg[0][index] = 1/t*(beacon_positions[0][index] - receiver_position[0])
    dg[1][index] = 1/t*(beacon_positions[1][index] - receiver_position[1])
    #print("dg-shapes", index, dg[0][index], dg[1][index])
  return dg


def read_position_from_csv(p_csv_file):
  beacon_positions = np.loadtxt(p_csv_file, delimiter=',')
  #beacon_positions_transpose = np.transpose(beacon_positions)
  #print(beacon_positions_transpose.shape)
  return beacon_positions


def read_distance_from_csv(d_csv_file):
  beacon_distances = np.loadtxt(d_csv_file, delimiter=',')
  #print(beacon_distances.shape)
  return beacon_distances

def calculate_moving_distance(bp_x, bp_y, receiver_state_vector, time):
  distance = np.sqrt(((receiver_state_vector[0] + receiver_state_vector[2]*time - bp_x)**2) + ((receiver_state_vector[1] + receiver_state_vector[3]*time - bp_y)**2))

  return distance
def g_moving(beacon_positions, receiver_state_vector, actual_distances, time):
  #print("BEACON HSHAPE", beacon_positions.shape)
  g = np.zeros(beacon_positions.shape[1])
  #print("G HSHAPE", g.shape)
  for index in range(g.shape[0]):
    g[index] = actual_distances[index] - calculate_moving_distance(beacon_positions[0][index], beacon_positions[1][index], receiver_state_vector, time)
  return g

def dg_moving(beacon_positions, receiver_state_vector, time):
  dg = np.zeros((beacon_positions.shape[1], receiver_state_vector.shape[0]))
  for index in range(beacon_positions.shape[1]):
    d = calculate_distance(beacon_positions[0][index], beacon_positions[1][index], receiver_state_vector)
    #print(beacon_positions)
    dg[0][index] = 1/d*(beacon_positions[0][index] - receiver_state_vector[0] - time*receiver_state_vector[2])
    dg[1][index] = 1/d*(beacon_positions[1][index] - receiver_state_vector[1] - time*receiver_state_vector[3])
    dg[2][index] = 1/d*(beacon_positions[0][index] - receiver_state_vector[0] - time*receiver_state_vector[2])*time
    dg[3][index] = 1/d*(beacon_positions[1][index] - receiver_state_vector[1] - time*receiver_state_vector[3])*time
  return dg

def extended_kalman_filter(beacon_positions, receiver_position, actual_distances_list, iterations, lamda, movement_type):
  if movement_type == "static":
    H = np.zeros((2,2))
    g_e =[]
    ite = []
    phi = [receiver_position]

    # generate 2D grid
    plt.figure(1, figsize=(7,7))
    plt.ylabel('Y coordinate ')
    plt.xlabel('X coordinate')

    x_grid, y_grid = np.mgrid[-5:25:0.01, -5:25:0.01]
    plt.xlim([-5, 25])
    plt.ylim([-5, 25])

    # draw contour lines of g
    p_1,p_2 = beacon_positions[0, :], beacon_positions[1, :]
    #x, y = phi[-1][0], phi[-1][1]
    f_all = np.zeros((4, *x_grid.shape))
    for i in range(0, 4):
        f_all[i, :, :] += actual_distances_list[99][i] - np.sqrt((x_grid - p_1[i])**2 + (y_grid - p_2[i])**2)

    f_norm = np.linalg.norm(f_all, axis=0)
    levels = 15
    plt.contour(x_grid, y_grid, f_norm, levels, label="Error")
    plt.plot(phi[-1][0], phi[-1][1], '*', markersize=10, color='red', label="phi coordinates")
    fig1 = plt.gcf()
    plt.legend()
    for i in range(0,beacon_positions.shape[1]):
      plt.plot(beacon_positions[0][i], beacon_positions[1][i], '+', markersize=10, color='blue', label="beacon-positions")
      plt.legend()
    for index in range(iterations):
      actual_distances = actual_distances_list[index]
      #print(actual_distances)
      for i in range(beacon_positions.shape[1]):
        phi_prev = phi[-1]
        g = compute_g(beacon_positions, phi_prev, actual_distances)
        dg = compute_dg(beacon_positions, phi_prev)
        #print("dg-shape", dg.shape)
        C = -dg.T
        #print("C-shape", C.shape)
        H = lamda*(H) + C.T@C
        #print("H-shape", H.shape)
        z = g.reshape(4, 1) - (dg.T@receiver_position).reshape(4, 1)
        #print("z-shape", z.shape)
        #print(phi[-1].shape, (z.reshape(4,1) - (C@(phi[-1])).reshape(4,1)).shape )
        #phi.append((phi_prev.reshape(2,1) + np.linalg.pinv( H )@( C.T )@( z.reshape(4,1) - (C@(phi_prev)).reshape(4,1) )).reshape(2,))
        phi.append((phi_prev.reshape(2,1) - np.linalg.pinv( H )@dg.reshape(2, 4)@g.reshape(4, 1)).reshape(2,))
        g_e.append(mean_error(g))
        ite.append(index)
        plt.plot(phi[-1][0], phi[-1][1], '*', markersize=10, color='red')
        fig1.canvas.draw()

        print(i, index, phi_prev)
        print("Error", mean_error(g))

    plt.figure(2, figsize=(7,7))
    #print(ite)
    plt.ylabel('Mean error ')
    plt.xlabel('Iterations')
    plt.plot(ite, g_e)



  elif movement_type == "dynamic":
    H = np.zeros((4,4))
    g_e =[]
    ite = []
    phi = [receiver_position]
    # generate 2D grid
    plt.figure(1, figsize=(7,7))
    plt.ylabel('Y coordinate ')
    plt.xlabel('X coordinate')

    x_grid, y_grid = np.mgrid[-5:25:0.01, -5:25:0.01]
    plt.xlim([-5, 25])
    plt.ylim([-5, 25])

    # draw contour lines of g
    p_1,p_2 = beacon_positions[0, :], beacon_positions[1, :]
    #x, y = phi[-1][0], phi[-1][1]
    f_all = np.zeros((4, *x_grid.shape))
    for i in range(0, 4):
        f_all[i, :, :] += actual_distances_list[99][i] - np.sqrt((x_grid - p_1[i])**2 + (y_grid - p_2[i])**2)

    f_norm = np.linalg.norm(f_all, axis=0)
    levels = 15
    plt.contour(x_grid, y_grid, f_norm, levels, label="Error")
    plt.plot(phi[-1][0], phi[-1][1], '*', markersize=10, color='red', label="phi coordinates")
    fig1 = plt.gcf()
    plt.legend()

    for index in range(iterations):
      actual_distances = actual_distances_list[index]
      print(actual_distances)
      for i in range(beacon_positions.shape[1]):
        phi_prev = phi[-1]
        g = g_moving(beacon_positions, phi_prev, actual_distances, index)
        dg = dg_moving(beacon_positions, phi_prev, index)
        #print("dg-shape", dg.shape)
        C = -dg.T
        #print("C-shape", C.shape)
        H = lamda*(H) + C.T@C
        #print("H-shape", H.shape)
        z = g.reshape(4, 1) - (dg.T@receiver_position).reshape(4, 1)
        #print("z-shape", z.shape)
        #print(phi[-1].shape, (z.reshape(4,1) - (C@(phi[-1])).reshape(4,1)).shape )
        #phi.append((phi_prev.reshape(2,1) + np.linalg.pinv( H )@( C.T )@( z.reshape(4,1) - (C@(phi_prev)).reshape(4,1) )).reshape(2,))
        phi.append((phi_prev.reshape(4,1) - np.linalg.pinv( H )@dg.reshape(4, 4)@g.reshape(4, 1)).reshape(4,))
        g_e.append(mean_error(g))
        ite.append(index)
        plt.plot((phi[-1][0] + index*phi[-1][2]), (phi[-1][1] + index*phi[-1][3]), '*', markersize=10, color='red')
        fig1.canvas.draw()
        print(i, index, phi_prev)
        print("Error", mean_error(g))

    plt.figure(2, figsize=(7,7))
    plt.ylabel('Mean error ')
    plt.xlabel('Iterations')
    #print(ite)
    plt.plot(ite, g_e)

  pass

def mean_error(g):
  mean = 0
  for i in g:
    mean = mean + abs(i/4)
  return mean

main()

