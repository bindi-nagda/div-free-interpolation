import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# B-splines of degree n
def B_old(x, n):
   if n == 0 and -0.5 <= x <= 0.5:
      return 1.0

   elif n > 0 and (-0.5)*(n+1) < x < 0.5*(n+1):
      c1 = ((n-1)/2+1-x) * B_old(x-0.5, n-1)
      c2 = ((n-1)/2+1+x) * B_old(x+0.5, n-1)
      s = (c1+c2)/n
      return s
   
   else:
      return 0.0

# x-component of vector field U_2a (note: U_2a is discretely divergence free)
def x_vel(x,y):
   u = np.sin(6*x+2)*np.sin(6*y+4)   
   return u

# y-component of vector field U_2a
def y_vel(x,y):
   v = np.cos(6*x+2)*np.cos(6*y+4)
   return v

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
      s = -1/6*pow((x-1),3)
   elif n == 3 and i == 1:
      s = 0.5*pow(x,3) + 2/3 - x**2
   elif n == 3 and i == 2:
      s = -1/2*pow(x,3) + 1/6 + 0.5*x**2 + 0.5*x
   elif n == 3 and i == 3:
      s = 1/6*pow(x,3)
   else:
      s = 0
   return s

def u_interp(x, y, dx, dy):
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_centers = xx[1::2] 
   y_centers = yy[1::2] 
   x_sides = xx[::2]   
   y_sides = yy[::2] 

   x1 = x_sides[x_sides < x].max()
   x2 = x_sides[x_sides > x].min()

   y1 = y_sides[y_sides < y].max()
   y2 = y_sides[y_sides > y].min()

   u = x_vel(x1-dx, y1-0.5*dx)*B(0, 3, x)*B(0, 2, y)\
   + x_vel(x1, y1-0.5*dx)*B(1, 3, x)*B(0, 2, y) \
   + x_vel(x2, y1-0.5*dx)*B(2, 3, x)*B(0, 2, y) \
   + x_vel(x2+dx, y1-0.5*dx)*B(3, 3, x)*B(0, 2, y) \
   + x_vel(x1-dx, y1+0.5*dx)*B(0, 3, x)*B(1, 2, y) \
   + x_vel(x1, y1+0.5*dx)*B(1, 3, x)*B(1, 2, y) \
   + x_vel(x2, y1+0.5*dx)*B(2, 3, x)*B(1, 2, y) \
   + x_vel(x2+dx, y1+0.5*dx)*B(3, 3, x)*B(1, 2, y) \
   + x_vel(x1-dx, y2+0.5*dx)*B(0, 3, x)*B(2, 2, y) \
   + x_vel(x1, y2+0.5*dx)*B(1, 3, x)*B(2, 2, y) \
   + x_vel(x2, y2+0.5*dx)*B(2, 3, x)*B(2, 2, y) \
   + x_vel(x2+dx, y2+0.5*dx)*B(3, 3, x)*B(2, 2, y) 

   return u

# Equation 10 in the paper. Note: dx and dy are input grid cell sizes      
def v_interp(x, y, dx, dy, c = 0):
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_centers = xx[1::2] 
   y_centers = yy[1::2] 
   x_sides = xx[::2]   
   y_sides = yy[::2]  

   x1 = x_sides[x_sides < x].max()
   x2 = x_sides[x_sides > x].min()

   y1 = y_sides[y_sides < y].max()
   y2 = y_sides[y_sides > y].min()

   v = y_vel(x1-0.5*dx, y1-dx)*B(0, 2, x)*B(0, 3, y)\
   + y_vel(x1-0.5*dx, y1)*B(0, 2, x)*B(1, 3, y) \
   + y_vel(x1-0.5*dx, y2)*B(0, 2, x)*B(2, 3, y) \
   + y_vel(x1-0.5*dx, y2+dx)*B(0, 2, x)*B(3, 3, y) \
   + y_vel(x1+0.5*dx, y1-dx)*B(1, 2, x)*B(0, 3, y) \
   + y_vel(x1+0.5*dx, y1)*B(1, 2, x)*B(1, 3, y) \
   + y_vel(x1+0.5*dx, y2)*B(1, 2, x)*B(2, 3, y) \
   + y_vel(x1+0.5*dx, y2+dx)*B(1, 2, x)*B(3, 3, y) \
   + y_vel(x2+0.5*dx, y1-dx)*B(2, 2, x)*B(0, 3, y) \
   + y_vel(x2+0.5*dx, y1)*B(2, 2, x)*B(1, 3, y) \
   + y_vel(x2+0.5*dx, y2)*B(2, 2, x)*B(2, 3, y) \
   + y_vel(x2+0.5*dx, y2+dx)*B(2, 2, x)*B(3, 3, y)    

   return v

# Bilinear interpolation for comparison
def bu_interp(x, y, dx, dy):
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_sides = xx[::2]   
   y_centers = yy[1::2] 

   x1 = x_sides[x_sides < x].max()
   x2 = x_sides[x_sides > x].min()

   y1 = y_centers[y_centers < y].max()
   y2 = y_centers[y_centers > y].min()

   fx_y1 = (x2-x)/(x2-x1)*x_vel(x1,y1) + (x-x1)/(x2-x1)*x_vel(x2,y1)
   fx_y2 = (x2-x)/(x2-x1)*x_vel(x1,y2) + (x-x1)/(x2-x1)*x_vel(x2,y2)
   u = (y2-y)/(y2-y1)*fx_y1 + (y-y1)/(y2-y1)*fx_y2
   return u

# Bilinear interpolation for comparison
def bv_interp(x, y, dx, dy):
   xx = np.arange(0,2+dx,dx)/2
   yy = np.arange(0,2+dy,dy)/2
   x_centers = xx[1::2] # cell-centers 
   y_sides = yy[::2]    # cell-sides 

   x1 = x_centers[x_centers < x].max()
   x2 = x_centers[x_centers > x].min()

   y1 = y_sides[y_sides < y].max()
   y2 = y_sides[y_sides > y].min()

   fy_x1 = (y2-y)/(y2-y1)*y_vel(x1,y1) + (y-y1)/(y2-y1)*y_vel(x2,y1)
   fy_x2 = (y2-y)/(y2-y1)*y_vel(x1,y2) + (y-y1)/(y2-y1)*y_vel(x2,y2)
   u = (x2-x)/(x2-x1)*fy_x1 + (x-x1)/(x2-x1)*fy_x2
   return u

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
   ax.plot(xx, [B_old(x, 0) for x in xx], 'r-', lw=2)
   ax.plot(xx, [B_old(x, 1) for x in xx], 'g-', lw=2)
   ax.plot(xx, [B_old(x, 2) for x in xx], 'b-', lw=2)
   ax.plot(xx, [B_old(x, 3) for x in xx], 'y-', lw=2)
   ax.plot(xx, [B_old(x, 4) for x in xx], 'k-', lw=2)
   ax.grid(True)
   plt.savefig('bsplines.png')

   dx = np.double(1/16) # input grid
   dy = np.double(1/16) # input grid
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
   x_coords =  np.random.uniform(low=0.05, high=0.95, size=(100,))  
   y_coords =  np.random.uniform(low=0.05, high=0.95, size=(100,))
   tolerance = 1e-8
   delta = 1e-6
   # x_coords = []
   # y_coords = []
   # # # Choose internal points on our refined 128 x 128 grid
   # for i in np.arange(1,128): 
   #    x_coords.append(i/128)   # Any coordinates that fall on the input grid result in a non-zero divergence.
   #    y_coords.append(i/128)
   for x in x_coords:
      for y in y_coords:  
         if abs(div(x,y,delta,dx,dy)) > tolerance:
            print(f"Divergence non-zero at {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")
         else:
            print(f"Divergence-free at cell {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")
  
   # Surface plots of original velocity field and interpolated velocity fields.
   dx = 1/32 # input grid
   dy = 1/32 # input grid
   xx = np.arange(2/64,1,1/64)
   yy = np.arange(2/64,1,1/64)
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
   surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

   # Add a color bar which maps values to colors.
   fig.colorbar(surf)
   plt.xlabel("X")
   plt.ylabel("Y")
   #plt.show()
   plt.savefig('u_interpolated.png')