"""
Example of use LVQ network
==========================

"""
import numpy as np
import neurolab as nl

# Create train samples
input = np.array([[-3, 0], [-2, 1], [-2, -1], [0, 2], [0, 1], [0, -1], [0, -2], 
                                                        [2, 1], [2, -1], [3, 0]])
target = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], 
                                                        [1, 0], [1, 0], [1, 0]])

# Create network with 2 layers:4 neurons in input layer(Competitive)
# and 2 neurons in output layer(liner)
net = nl.net.newlvq(nl.tool.minmax(input), 4, [.6, .4])
# Train network
error = net.train(input, target, epochs=1000, goal=-1)

# Plot result
import pylab as pl
xx, yy = np.meshgrid(np.arange(-3, 3.4, 0.2), np.arange(-3, 3.4, 0.2))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
i = np.concatenate((xx, yy), axis=1)
o = net.sim(i)
grid1 = i[o[:, 0]>0]
grid2 = i[o[:, 1]>0]

class1 = input[target[:, 0]>0]
class2 = input[target[:, 1]>0]

pl.plot(class1[:,0], class1[:,1], 'bo', class2[:,0], class2[:,1], 'go')
pl.plot(grid1[:,0], grid1[:,1], 'b.', grid2[:,0], grid2[:,1], 'gx')
pl.axis([-3.2, 3.2, -3, 3])
pl.xlabel('Input[:, 0]')
pl.ylabel('Input[:, 1]')
pl.legend(['class 1', 'class 2', 'detected class 1', 'detected class 2'])
pl.show()