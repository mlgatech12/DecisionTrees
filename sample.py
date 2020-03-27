import numpy as np
import random
mydict = {"a" : 1, "b" : 2, "c" : 3}
for key, value in mydict.items():
    print(key ,  ", " , value)
    
print (sum(mydict.values()))

X = np.array(([21, 22, 23, 91, 10, 20], 
              [11, 22, 33, 56, 11, 30], 
              [43, 77, 89, 45, 12, 53], 
              [430, 770, 890, 450, 120, 530],
              [231, 881, 851, 401, 201, 541],
              [232, 882, 852, 402, 202, 542],
              [233, 883, 853, 403, 203, 543]))
a = np.array([1, 2, 3, 4, 5, 6, 7])

print(a[0])
y = np.array([a])

X_new = np.concatenate((X, y.T), axis=1)

print("length = ", len(X_new))

for i in range(0, (X.shape[1])): #iterate through all columns except one
    threshold = np.mean(X[:,i])
    print("mean =", threshold)
    left_set = X_new[X_new[:,i] <= threshold]
    left_class = left_set[:,-1].tolist()
            
    right_set = X_new[X_new[:,i] > threshold]
    right_class = right_set[:,-1].tolist()
            
max_index = random.randint(0,3)
print("best index = ", max_index )
threshold = np.mean(X[:,max_index])
print("mean =", threshold)
left_set = X_new[X_new[:,max_index] <= threshold]
left_class = left_set[:,-1]
print("left_features =", left_set[:,0:-1])
print("left_class =", left_class)
            
right_set = X_new[X_new[:,max_index] > threshold]
right_class = right_set[:,-1]
print("right_features =", right_set[:,0:-1])
print("right_class =", right_class)
       

#class_1 = left[:,-1].tolist()
#print(class_1)

#print(new.shape[1])
#print(left)

#left_class = np.array([1. , 1.])

#print(len(np.unique(left_class)))
#print(left_class[0])



#u[np.argmax(np.bincount(indices))]

#print(u.max())
split_arr =[]
total = 25
k = 6

"""
if(len(X_new) % k == 0):
    print("here")
    split_arr = np.split(X_new, k, axis=0)
else:
    t = int(len(X_new) / k)
    #t = int(28 / k)
    t_arr = []
    prev = 0
    for i in range(1,k):
        t_arr.append(t+prev)
        prev = prev+t
    print("split array param = ", t_arr)
    split_arr = np.split(X_new, [2,4], axis=0)   


t = 25 - (25 // k)
"""

split_arr = np.split(X_new, [2], axis=0)   

print("split array =", split_arr)
print("feature =", split_arr[0][:,0:-1])
print("classes =", split_arr[0][:,-1])

chk1 = X_new[0:0, 0:-1]
#print(chk1)
chk2 = X_new[5:, 0:-1]
#print(chk2)

train_features = np.concatenate((chk1, chk2))
print(train_features)

train_classes = np.concatenate((X_new[0:0, -1] , X_new[5:, -1]))
print(train_classes)

tr_classes = np.array([train_classes])

train = np.concatenate((train_features, tr_classes.T), axis=1)
print(len(train.tolist()))      

"""       
8,8,9

[2,2,3]
[2,4] 
"""
    
attributes_ct = 3
attribs = np.random.choice(X_new.shape[1], attributes_ct, replace=False)

print(attribs)
p = np.array([X_new[:,attribs[0]]]).T
q = np.array([X_new[:,attribs[1]]]).T


print(np.hstack((p,q)))


my_arr = [2, 4, 1, 4, 3, 3, 4, 1]
majority = np.argmax(np.bincount(np.array(my_arr)))

print(majority)


test = np.array([0., 1., 1., 1., 1., 1., 1.])

test = test.astype(int)

frequent_val = np.argmax(np.bincount(test))

print(float(frequent_val))


#random sampling
num_rows_2_sample = 4
sam = X_new[np.random.choice(X_new.shape[0], num_rows_2_sample, replace=False)]
print(sam)


class_label_1 = [0, 0, 3, 4, 5]
class_label_2 = [1, 0, 3, 4, 0]
class_label_3 = [1, 0, 2, 4, 0]
t_label = []

t_label.append(class_label_1)
t_label.append(class_label_2)
t_label.append(class_label_3)

t_arr = np.array(t_label)
t_arr = t_arr.T

def majority(arr):
    return float(np.argmax(np.bincount(arr)))
    
    

new_t = np.apply_along_axis(majority, 1, t_arr)

print(new_t)

    

            
          

