import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# B-splines of degree n
def B(x, n):
   if n == 0 and -0.5 <= x <= 0.5:
      return 1.0

   elif n > 0 and (-0.5)*(n+1) < x < 0.5*(n+1):
      c1 = ((n-1)/2+1-x) * B(x-0.5, n-1)
      c2 = ((n-1)/2+1+x) * B(x+0.5, n-1)
      s = (c1+c2)/n
      return s
   
   else:
      return 0.0


# x-component of vector field U_2a (note: U_2a is discretely divergence free)
def x_vel(x,y):
   u = np.sin(x+2)*np.sin(y+4)   
   return u

# y-component of vector field U_2a
def y_vel(x,y):
   v = np.cos(x+2)*np.cos(y+4)
   return v

# Equation 9 in the paper. Note: dx and dy are input grid cell sizes
def u_interp_old(x, y, dx, dy):
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_centers = xx[1::2] 
   y_centers = yy[1::2] 
   x_sides = xx[::2]   
   u = 0.0
   for i in range(len(x_centers)):
      for j in range(len(y_centers)):
         x_sides = i*dx
         y_centers = (j+1/2)*dy
         u += (x_vel(x_sides, y_centers)*B((x - x_sides)/dx, 3)*B((y - y_centers)/dy, 2))
   # for x_sides in np.arange(x - 2*dx, x + 2*dx):
   #    for y_centers in np.arange(y - 2*dy, x + 2*dy):
   #       u += (x_vel(x_sides, y_centers)*B(x - x_sides)/dx, 3)*B((y - y_centers)/dy, 2)
   return u

######################################
# B-splines of degree n
def B(i, n, x):
   if n == 0 and i == 0:
      s = 1
   elif n == 1 and i == 0:
      s = 1 - x
   elif n == 1 and i == 1:
      s = x
   elif n == 2 and i == 0:
      s = 0.5*(x-1)**2   
   elif n == 2 and i == 1:
      s = -x**2 + x + 0.5
   elif n == 2 and i == 2:
      s = 0.5*x**2
   elif n == 3 and i == 0:
      s = -1/6*np.power((x-1),2)
   elif n == 3 and i == 1:
      s = 0.5*np.power(x,3) + 2/3 - x**2
   elif n == 3 and i == 2:
      s = -1/2*np.power(x,3) + 1/6 + 0.5*x**2 + 0.5*x
   elif n == 3 and i == 3:
      s = 1/6*np.power(x,3)
   else:
      s = 0
   return s

def u_interp(x, y, dx, dy):
   u = 0.0
   for i in range(4):
      for j in range(3):
         xval = i*dx
         yval = j*dy
         u += x_vel(xval, yval)*B(i, 2, x)*B(j, 1, y)
   # for x_sides in np.arange(x - 2*dx, x + 2*dx):
   #    for y_centers in np.arange(y - 2*dy, x + 2*dy):
   #       u += (x_vel(x_sides, y_centers)*B(x - x_sides)/dx, 3)*B((y - y_centers)/dy, 2)
   return u

# Equation 10 in the paper. Note: dx and dy are input grid cell sizes      
def v_interp(x, y, dx, dy):
   v = 0.0
   # for i in range(len(x_centers)):
   #    for j in range(len(y_centers)):
   #       v = v + (y_vel(x_centers[i], y_sides[j+1])*B((x - x_centers[i])/dx, 2)*B((y - y_sides[j+1])/dy, 3))
   for j in range(4):
      for i in range(3):
         xval = i*dx
         yval = j*dy
         v += y_vel(xval, yval)*B(i, 1, x)*B(j, 2, y)      
   return v

# Equation 17 in the paper
def div_17(x, y, dx, dy): 
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_centers = xx[1::2] 
   y_centers = yy[1::2]
   y_sides = yy[::2]
   d = 0.0
   for i in range(len(x_centers)):
      for j in range(len(y_centers)):
         d = (x_vel(x_sides[i+1], y_centers[j]) - x_vel(x_sides[i], y_centers[j]))/dx \
            + (y_vel(x_centers[i], y_sides[j+1]) - y_vel(x_centers[i], y_sides[j]))/dy
         d = d*B((x - x_centers[i])/dx, 2)*B((y - y_centers[j])/dy, 2)
   return d

if __name__=="__main__":

   # # Plot some B-splines
   # fig, ax = plt.subplots()
   # xx = np.linspace(-4.0, 4.0, 400)
   # ax.plot(xx, [B(x, 0) for x in xx], 'r-', lw=2)
   # ax.plot(xx, [B(x, 1) for x in xx], 'g-', lw=2)
   # ax.plot(xx, [B(x, 2) for x in xx], 'b-', lw=2)
   # ax.plot(xx, [B(x, 3) for x in xx], 'y-', lw=2)
   # ax.plot(xx, [B(x, 4) for x in xx], 'k-', lw=2)
   # ax.grid(True)
   # plt.savefig('bsplines.png')

   dx = np.double(1/32) # input grid
   dy = np.double(1/32) # input grid
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_sides = xx[::2]    # cell-sides 
   x_centers = xx[1::2] # cell-centers 
   y_sides = yy[::2]    # cell-sides 
   y_centers = yy[1::2] # cell-centers 

   # divergence of cell (i,j) in input grid 
   def div_original(i,j):
      d = (x_vel(x_sides[i+1], y_centers[j]) - x_vel(x_sides[i], y_centers[j]))/dx \
         + (y_vel(x_centers[i], y_sides[j+1]) - y_vel(x_centers[i], y_sides[j]))/dy
      return d
   
   # Compute a second order accurate approximation of the divergence at test location (x,y)
   # by sampling the interpolated fields
   def div(x, y, delta, dx, dy): 
      d = (u_interp(x+delta, y, dx, dy) - u_interp(x-delta, y, dx, dy))/(2*delta) \
         + (v_interp(x, y+delta, dx, dy) - v_interp(x, y-delta, dx, dy))/(2*delta)
      return d

   # # Randomly sample the computational domain to generate test coordinates
   # x_coords =  np.random.uniform(low=2*dx, high=1.0-2*dx, size=(64,))  
   # y_coords =  np.random.uniform(low=2*dx, high=1.0-2*dx, size=(64,))
   tolerance = 1e-8
   delta = 1e-6
   x_coords = []
   y_coords = []
   # Choose internal points on our refined 128 x 128 grid
   # for i in np.arange(1,128): 
   #    x_coords.append(i/128)   # Any coordinates that fall on the input grid result in a non-zero divergence. Why? Surface plot below provides a little insight.
   #    y_coords.append(i/128)
   # for x in x_coords:
   #    for y in y_coords:  
   #       if div(x,y,delta,dx,dy) > tolerance:
   #          print(f"Divergence non-zero at {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")
   #       else:
   #          print(f"Divergence-free at cell {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")
  
   # Surface plots of original velocity field and interpolated velocity fields.
   dx = 1/32 # input grid
   dy = 1/32 # input grid
   xx = np.arange(0,1,1/64)
   yy = np.arange(0,1,1/64)
   zz = []

   for x in xx:
      for y in yy:
         #zz.append(x_vel(x,y))
         zz.append(u_interp(x,y,dx,dy))

   fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

   # Make data.
   X, Y = np.meshgrid(xx, yy)
   Z = np.array(zz)
   Z = np.reshape(zz,[xx.shape[0],yy.shape[0]])

   # Plot the surface.
   surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

   # Add a color bar which maps values to colors.
   fig.colorbar(surf)

   plt.show()
   plt.savefig('u_interpolated.png')