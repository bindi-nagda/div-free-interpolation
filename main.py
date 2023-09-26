import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# B-splines of degree n
def B(x, n):
   if n == 0 and -0.5 <= x <= 0.5:
      return 1.0

   if  (-0.5)*(n+1) < x < 0.5*(n+1):
      c1 = ((n-1)/2+1-x)/n * B(x-0.5, n-1)
      c2 = ((n-1)/2+1+x)/n * B(x+0.5, n-1)
      s = c1+c2
      return s

   return 0

# x-component of vector field U_2a (note: U_2a is discretely divergence free)
def x_vel(x,y):
   u = np.sin(370*x+2)*np.sin(370*y+4)   
   return u

# y-component of vector field U_2a
def y_vel(x,y):
   v = np.cos(370*x+2)*np.cos(370*y+4)
   return v

# Equation 9 in the paper. Note: dx and dy are input grid cell sizes
def u_interp(x, y, dx, dy):
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_centers = xx[1::2] 
   y_centers = yy[1::2] 
   x_sides = xx[::2]   
   u = 0.0
   for i in range(len(x_centers)):
      for j in range(len(y_centers)):
         u = u + x_vel(x_sides[i+1], y_centers[j])*B((x - x_sides[i+1])/dx, 3)*B((y - y_centers[j])/dy, 2)
   return u

# Equation 10 in the paper. Note: dx and dy are input grid cell sizes      
def v_interp(x, y, dx, dy):
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_centers = xx[1::2] 
   y_centers = yy[1::2]
   y_sides = yy[::2]
   v = 0.0
   for i in range(len(x_centers)):
      for j in range(len(y_centers)):
         v = v + y_vel(x_centers[i], y_sides[j+1])*B((x - x_centers[i])/dx, 2)*B((y - y_sides[j+1])/dy, 3)
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

   # Plot some B-splines
   fig, ax = plt.subplots()
   xx = np.linspace(-4.0, 4.0, 400)
   ax.plot(xx, [B(x, 0) for x in xx], 'r-', lw=2)
   ax.plot(xx, [B(x, 1) for x in xx], 'g-', lw=2)
   ax.plot(xx, [B(x, 2) for x in xx], 'b-', lw=2)
   ax.plot(xx, [B(x, 3) for x in xx], 'y-', lw=2)
   ax.plot(xx, [B(x, 4) for x in xx], 'k-', lw=2)
   ax.grid(True)
   plt.savefig('bsplines.png')

   dx = np.double(1/64) # input grid
   dy = np.double(1/64) # input grid
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

   # Randomly sample the computational domain to generate test coordinates
   x_coords =  np.random.uniform(low=2*dx, high=1.0-2*dx, size=(64,))  
   y_coords =  np.random.uniform(low=2*dx, high=1.0-2*dx, size=(64,))
   tolerance = 1e-8
   delta = 1e-6
   x_coords = []
   y_coords = []
   # Choose internal points on our refined 128 x 128 grid
   for i in np.arange(21,100): 
      x_coords.append(i/128)   # Any coordinates that fall on the input grid result in a non-zero divergence. Why? Surface plot below provides a little insight.
      y_coords.append(i/128)
   for x in x_coords:
      for y in y_coords:  
         if div(x,y,delta,dx,dy) > tolerance:
            print(f"Divergence non-zero at {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")
         else:
            print(f"Divergence-free at cell {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")
  
   # Surface plots of original velocity field and interpolated velocity fields.
   dx = 1/8 # input grid
   dy = 1/8 # input grid
   xx = np.arange(0,1,1/128)
   yy = np.arange(0,1,1/128)
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

   plt.savefig('u_interpolated.png')
   

