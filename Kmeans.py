import numpy as np
import pandas as pd
import math

def read_input():
	file = pd.read_csv(r'C:\Users\tai91\OneDrive\Desktop\working folder\ML\mnist_test.csv')
	data = file[0:100]
	return data

def cluster_devide(arr_input,current_cluster):
    x_shape,y_shape = arr_input.shape
    temp_cluster = []
    for i_index in range(x_shape):
        # go through cluster points
        # min distance always < y_shape because i have normed the array input
        # Assign min distance to y_shape
        min_distance = y_shape
        index_value = -1 
        for j_index in range(10):
            temp_distance = 0.00000
            # go through pixels in each point
            for k_index in range(y_shape):
                temp_distance = temp_distance + (arr_input[i_index][k_index] - current_cluster[j_index][k_index])**2
            temp_distance = math.sqrt(temp_distance)
            #print(temp_distance)
            if min_distance > temp_distance:
                index_value = j_index
                min_distance = temp_distance
        temp_cluster.append(index_value)
    return   temp_cluster 

def cluster_group(arr_input,temp_label,cluster_num):
    x_shape,y_shape = arr_input.shape
    temp_cluster = []
    cnt_num = [0]*cluster_num
    for i in range(cluster_num):
        temp_cluster.append([0]*y_shape)
        for j in range(x_shape):
            if temp_label[j] == i:
                temp_cluster[i] = temp_cluster[i] + arr_input[j]
                cnt_num[i] = cnt_num[i] + 1
    temp_cluster = np.array(temp_cluster)
    cnt_num = np.array(cnt_num)
    for i in range(cluster_num):
        temp_cluster[i] = temp_cluster[i]/cnt_num[i]
    cluster_update = [0]*cluster_num
    for i in range(cluster_num):
        min_distance = y_shape
        sum_distance = 0
        index = -1
        for j in range(x_shape):
            sum_distance =  np.sum((temp_cluster[i]-arr_input[j])**2)
            if min_distance > sum_distance : 
                min_distance = sum_distance
                index = j
        cluster_update[i] = arr_input[j]
    #print(cluster_update[0] - cluster_update[1])
    return cluster_update        
            
    return temp_label
def Calc_Error(temp_label,y_label):
    return 1

# this is main functon of Kmean cluster
def K_mean(arr_input,y_label,cluster_num):

    # find initial cluster group
    ''' for i in range(10):
        j = 0
        while(y_label[j] != i):
            j = j + 1
        #print(j)
        Cluster_init.append(arr_input[j])
    '''
    # choose init cluster group
    # I choose first cluster number of array input 
    Cluster_init = arr_input[0:cluster_num]

    # x_size for number of elements in input array, y_size is lenght of each element
    x_size,y_size = arr_input.shape

    cluster_update = arr_input[-10:]
    # label_update for every update the cluster group
    label_update = [0] * x_size
    run_cnt = 0
    # update until cluster_init == cluster_update
    while np.array_equal(Cluster_init,cluster_update) != True:
        Cluster_init = cluster_update
        # assign label of element to the nearest cluster
        label_update = cluster_devide(arr_input,Cluster_init)
        # update the center by choose the element which is the nearest point of average of each cluster
        cluster_update = cluster_group(arr_input,label_update,cluster_num)
        #run_cnt = run_cnt + 1


    # assign the label of cluster to group cluster
    label_result = label_update
    error_rate = Calc_Error(label_result,y_label)
    return  label_update,error_rate

    
	
test = read_input()

# preprocess to preapre array
x,y = test.shape
y_label = np.array(test.label)
x_point = np.array(test.iloc[0:,1:])
x_point = x_point / 255

# Call K_mean to define label of input x_point, 10 clusters of NMist dataset
label_cal = K_mean(x_point,y_label,10)
#print(label_cal)
# Calculate error rate 
#error_rate = Error_cal(y_label,label_cal)
#print(x_point)
