# import librarys: 
#-------------------------------------------------------------
import numpy as np                      #use in mathematical equation
import pandas as pd                     #use in statical data
import matplotlib.pyplot as plt         #draw graphs scatter and matplot contour
from matplotlib import cm               #use in Creating Colormaps in Matplotlib
#---------------------------------------------------------------
# LOAD regression_data.csv to dataframe pandas

path = 'c://users//Mostafa Mahmoud//regression_data.csv'
data = pd.read_csv(path, header=None, names=['input' , 'output'])

# display first samples of data and see the describe

print('#################################')
print('data = \n' , data.head(20))
print('data,describe = \n' , data.describe())
print('#################################')
#-------------------------------------------------------------------
#plot the data as scatterplot

data.plot(kind='scatter', x='input', y='output', figsize=(4,4))

# build matrix X with the input data and add the x0 values =1
data.insert(0, 'Ones', 1)
print('new data = \n' , data.head(20))
print('#################################')

# split the data to x(input) and y(real result)
cols = data.shape[1]                 # 97 row * 3 colums 
X = data.iloc[ : , 0 : cols-1]       # column 0 (ones) , column 1 {input}
y = data.iloc[ : , cols-1 : cols ]   # column 2 {output}

print('**************************************')
print('X data = \n' ,X.head(20) )
print('y data = \n' ,y.head(20) )
print('**************************************')

#-------------------------------------------------------------------

# initialize the theta vector with zeros and make x,y as matrix

X = np.matrix(X.values)                # convert from data into numpy matrices 
y = np.matrix(y.values)
theta = np.matrix(np.array(np.repeat(0,cols-1)))

print('#################################')
print('X \n',X)
print('X shape = ',X.shape)
print('theta \n',theta)
print('theta.shape =  \n',theta.shape)
print('y \n',y)
print('y shape = ',y.shape)
print('#################################################')

#----------------------------------------------------------------------
# Root Mean Square as cost function report of cost when theta is zeros

def computeCost(X, y, theta):                 #compute cost function when theta= 0`s j(theta)  
    z = np.power(((X * theta.T) - y), 2)      # j(theta0 , theta1 ) = 1/2m summation (h.theta (x^i)-y^i)^2  z => h(theta)
    return np.sum(z) / (2 * len(X))           # Goal cost func minmize j(theta0 , theta 1) 

print('computeCost(x, y, theta) = ' , computeCost(X, y, theta))
print('################################################')
#-------------------------------------------------------------------------------
# The batch gradient descent.
def gradientDescent(X, y, theta, alpha, iters):                 #2 for loop in BGD {Error - temporary theta in 1500 iteration}
    temporary_theta = np.matrix(np.zeros(theta.shape))#(1,2) of 0
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)#(1,1500)
    theta1 = np.zeros(iters)#(1,1500)
    theta2 = np.zeros(iters)#(1,1500)

    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temporary_theta[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temporary_theta                # temporary of theta
        theta1[i]=theta[0,0]
        theta2[i]=theta[0,1]
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost ,theta1, theta2
#------------------------------------------------------------------------------------
# learning rate is 0.01 and epochs 1500 report the final theta you have 
#initialize variables for learning rate and epochs
alpha = 0.01        #learning rate
iters = 1500        #is the number of batches needed to compelet one epochs

# perform linear regression on the data set
g, cost ,theta1,theta2= gradientDescent(X, y, theta, alpha, iters)
print('            The final report\n')
print('g = ' , g)
print('theta1  = ' , theta1[0:5] )
print('theta2  = ' , theta2[0:5] )
print('cost  = '   , cost[0:5] )
print('computeCost = ' , computeCost(X, y, g))   
#----------------------------------------------------------------------------------------
# Plot the data as scatter and the model as line.
x = np.linspace(data.input.min(), data.input.max(), 100)
#print('x \n',x)
#print('g \n',g)    # theta  

f = g[0, 0] + (g[0, 1] * x)        # hypothise=>hx (g)=TH1+TH2 * x
#print('f \n',f)
#-------------------------------------------------------------------------------------------
# draw the line for predicted output vs. input size

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'g', label='Prediction')
ax.scatter(data.input, data.output, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('input')
ax.set_ylabel('output')
ax.set_title('predicted output vs. input size')
#------------------------------------------------------------------------------------------
# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'g')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
#-----------------------------------------------------------------------------------------
# Given input = [1, 3.5] and predict the out put
new_input=np.array([1,3.5])
print("New_input = ",new_input)
new_input *g.T
#-----------------------------------------------------------------------------------------
#plot a contour-plot given your thetas and the cost function J

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
theta1, theta2 = np.mgrid[-1:1:30j, -1:1:30j]
cost = np.sin(np.pi*theta1)*np.sin(np.pi*theta2)
ax.plot_surface(theta1, theta2, cost, cmap="autumn_r", lw=0.5, rstride=1, cstride=1, alpha=0.01)
ax.contour(theta1, theta2, cost, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
ax.contour(theta1, theta2, cost, 10, lw=3, colors="g", linestyles="solid")
plt.show()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(theta1, theta2, cost , 50 , cmap=cm.cool)
ax.set_xlabel('theta1')
ax.set_ylabel('theta2')
ax.set_zlabel('cost')
ax.set_title('3D contour for cosine')
plt.show()

#------------------------------------------------------------------------------------------ 





