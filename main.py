import matplotlib.pyplot as plt
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
   u = np.sin(x+2)*np.sin(y+4)   
   return u

# y-component of vector field U_2a
def y_vel(x,y):
   v = np.cos(x+2)*np.cos(y+4)
   return v

# Equation 9 in the paper
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

# Equation 10 in the paper      
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

   dx = 1/32
   dy = 1/32
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
   x_coords =  np.random.uniform(low=0.1, high=0.9, size=(50,))   # Points near the domain boundary result in non-zero divergence
   y_coords =  np.random.uniform(low=0.1, high=0.9, size=(50,))
   tolerance = 1e-9
   delta = 1e-6
   for x in x_coords:
      for y in y_coords:  
         if div(x,y,delta,dx,dy) > tolerance:
            print(f"Divergence non-zero at {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")
         # else:
         #    print(f"Divergence-free at cell {(x,y)}. Divergence = {div(x,y,delta,dx,dy)}")

