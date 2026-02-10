# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 00:24:05 2026

@author: OMIDGHNV
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random
plt.ion() 



################################################################################################
####################   Sector 1 : Grid Generation   #########################################
################################################################################################

R_out = 9
R_in = 1

N_r = 81
N_tetha = 81

d_r =  (R_out - R_in) / (N_r-1)
d_tetha = math.pi / (N_tetha-1)

x_cordinate_mat = np.zeros((N_r,N_tetha))
y_cordinate_mat = np.zeros((N_r,N_tetha))

for i in range(0,N_r):
    
    for j in range(0,N_tetha):
        
        x = (R_in + i*d_r) * math.cos(j*d_tetha)
        y = (R_in + i*d_r) * math.sin(j*d_tetha)
        
        x_cordinate_mat[i,j] = x
        y_cordinate_mat[i,j] = y
        
plt.figure(1,figsize=(10,10))
plt.scatter(x_cordinate_mat, y_cordinate_mat, s=10, c='blue', alpha=0.5)
plt.axis('equal')
plt.grid(True)
plt.show()


#########################################################################################################################################################
###############################     Sector 2 : Calculating Potential     ##########################################
#########################################################################################################################################################


#Flow information 

desired_iteration = 500
R_ref = R_in

omega = 1.5
gama = 1.4
R = 288
T_ambient = 298.15
a_inf = math.sqrt(T_ambient * R * gama)
M_inf = 0.2          # for mach=0 , put small quantity instead of 0 , for example 1e-5
u_inf = M_inf * a_inf


#initial data for node's (tow mode available)
Phi_matrix = np.zeros((N_r,N_tetha))
for i in range(0,N_r):
    
    for j in range(0,N_tetha):
        
        #mode 1 (exact solotion of potential eq for incompresible flow)
        # x_i = (R_in + i*d_r)/R_ref
        # tetha = j * d_tetha
        # phi_i = (x_i + (1/x_i)) *math.cos(tetha)
        # Phi_matrix[i,j] = phi_i
        
        #mode 2
        x_i = (R_in + i*d_r)/R_ref
        tetha = j * d_tetha
        phi_i = x_i *math.cos(tetha)
        Phi_matrix[i,j] = phi_i
        
 
        
# iterating loop for updating potential matrix

counter = 0 
d_x = d_r
for iteration in range(desired_iteration):
    counter = counter +1
    print('iteration %s of %s'%(counter,desired_iteration))
    #Assigning Bc's
    Phi_matrix[0,:] = Phi_matrix[1,:]     #Radial Boundary condition in r-directon on the surface of the cylinder where gradient is zero in r-direction
    Phi_matrix[:,0] = Phi_matrix[:,1]     #Gradient in tetha direction is zero in 0 and pi angle
    Phi_matrix[:,N_tetha-1] = Phi_matrix[:,N_tetha-2]
    
    for i in range(1,N_r-1):
        
        x_i = (R_in + i*d_r) / R_ref
        
        for j in range(1,N_tetha - 1):
            
            phi_x = (Phi_matrix[i+1,j]-Phi_matrix[i-1,j]) / (2*d_x)
            phi_tetha = (Phi_matrix[i,j+1]-Phi_matrix[i,j-1]) / (2*d_tetha)
            phi_x_thet = (Phi_matrix[i+1,j+1]-Phi_matrix[i+1,j-1]-Phi_matrix[i-1,j+1]+Phi_matrix[i-1,j-1]) / (4*d_x*d_tetha)
            
            c = 1 + (  (0.5 * (gama-1) * M_inf**2)*(1 - phi_x**2 - (phi_tetha**2/x_i**2) )  )
            A = c**2 - (M_inf**2 * phi_x**2)
            B = c**2 - (M_inf**2 / x_i**2)*(phi_tetha**2)
            C = c**2 + (M_inf**2 / x_i**2)*(phi_tetha**2)
            D = 2*(M_inf**2 / x_i**2)*(phi_x)*(phi_tetha)*(phi_x_thet)
            
            phi_new = ( (A/d_x**2)*(Phi_matrix[i+1,j]+Phi_matrix[i-1,j])  +  ( (B/(x_i**2 * d_tetha**2)) *  (Phi_matrix[i,j+1]+Phi_matrix[i,j-1]) )   +  (C/(2*x_i*d_x))*(Phi_matrix[i+1,j]-Phi_matrix[i-1,j])  - D) * ( 1/( (2*A / d_x**2) + (2*B /( x_i**2 * d_tetha**2)) ) ) 
            
            Phi_matrix[i,j] = omega*phi_new + (1-omega)*Phi_matrix[i,j]



plt.figure(2,figsize=(10, 8))
contour = plt.contourf(x_cordinate_mat, y_cordinate_mat, Phi_matrix, 
                       levels=50, cmap='jet')
contour_lines = plt.contour(x_cordinate_mat, y_cordinate_mat, Phi_matrix, 
                           levels=50,  
                           colors='black',  
                           linewidths=1.5,   
                           alpha=0.7) 
plt.colorbar(contour, label='φ (Potential)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour of Velocity Potential φ')
plt.axis('equal')
plt.grid(True, alpha=0.3)






####################################################################################################################################################
####### Sector 3 :Calculating flow parameter over cylinder Surface and plotting them  #########################
####################################################################################################################################################

phi_surface = Phi_matrix[0,:]

u_surface = np.zeros(len(phi_surface))

for i in range(0,len(phi_surface)):
    if i == 0 :
        u_surface[i] = 0
    elif i == len(phi_surface) - 1:
        u_surface[i] = 0
    else :
        u_surface[i] = u_inf*((phi_surface[i+1] - phi_surface[i-1] ) / (2*d_tetha))
        

        
T_surrface = np.zeros(len(phi_surface))

for i in range(0,len(T_surrface)):
    T_surrface[i] = T_ambient + (0.5*(u_inf**2) - 0.5*(u_surface[i]**2)) / ((gama*R)/(gama-1))

a_surrface = np.zeros(len(phi_surface))

for i in range(0,len(a_surrface)):
    a_surrface[i] = math.sqrt(T_surrface[i]*gama*R)
    
    
M_surrface = np.zeros(len(phi_surface))   

for i in range(0,len(a_surrface)):
    M_surrface[i] = abs(u_surface[i] / a_surrface[i])
    

Cp_surface = np.zeros(len(phi_surface))

for i in range(len(phi_surface)):
    p_ratio = ((1 + (gama-1)/2 * M_inf**2) / 
               (1 + (gama-1)/2 * M_surrface[i]**2))**(gama/(gama-1))
               
    Cp_surface[i] = (2 / (gama * M_inf**2)) * (p_ratio - 1)  




###plot exact solotion beside code calculation's for comparison

Phi_exact = np.zeros((N_r,N_tetha))

for i in range(0,N_r):
    
    for j in range(0,N_tetha):
        
        # mode 1
        x_i = (R_in + i*d_r)/R_ref
        tetha = j * d_tetha
        phi_i = (x_i + (1/x_i)) *math.cos(tetha)
        Phi_exact[i,j] = phi_i
        

error = np.abs(Phi_matrix - Phi_exact)
max_error = np.max(error)
mean_error = np.mean((error))


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cont1 = axes[0].contourf(x_cordinate_mat, y_cordinate_mat, Phi_matrix, levels=40, cmap='jet', alpha=0.7)
contour_lines1 = axes[0].contour(x_cordinate_mat, y_cordinate_mat, Phi_matrix, levels=50, colors='black', linewidths=0.8, alpha=0.9)
axes[0].set_title('Code Solution ($\phi$)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].axis('equal')
plt.colorbar(cont1, ax=axes[0], label='$\phi$')

cont2 = axes[1].contourf(x_cordinate_mat, y_cordinate_mat, Phi_exact, levels=40, cmap='jet', alpha=0.7)
contour_lines2 = axes[1].contour(x_cordinate_mat, y_cordinate_mat, Phi_exact, levels=50, colors='black', linewidths=0.8, alpha=0.9)
axes[1].set_title('Exact Solution ($\phi$)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].axis('equal')
plt.colorbar(cont2, ax=axes[1], label='$\phi$')

cont3 = axes[2].contourf(x_cordinate_mat, y_cordinate_mat, error, levels=50, cmap='hot')
axes[2].set_title(' Error ')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].axis('equal')
plt.colorbar(cont3, ax=axes[2])

plt.tight_layout()
plt.show()




# plotting Cp_surface in respect to tetha

if M_inf < 0.005 :

    theta_surface = np.linspace(0, np.pi, N_tetha)
    Cp_exact = Cp_exact = 1 - 4 * np.sin(theta_surface)**2
    Cp_prandtl_glaurt = Cp_exact / np.sqrt(1 - M_inf**2)
    
    plt.figure(figsize=(12, 8))
    plt.plot(theta_surface, Cp_surface, 'b-', linewidth=2, label=f'code ($M_\infty$={M_inf})')
    plt.plot(theta_surface, Cp_exact , 'r-' , linewidth=2, label=f'Exact(incompresible) ($M_\infty$={M_inf})')
    plt.xlabel(r'$\theta$ (rad)')
    plt.ylabel(r'$C_p$')
    plt.title('Pressure Coefficient on Cylinder Surface')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    theta_surface = np.linspace(0, np.pi, N_tetha)
    Cp_exact = Cp_exact = 1 - 4 * np.sin(theta_surface)**2
    Cp_prandtl_glaurt = Cp_exact / np.sqrt(1 - M_inf**2)
    
    plt.figure(figsize=(12, 8))
    plt.plot(theta_surface, Cp_surface, 'b-', linewidth=2, label=f'code($M_\infty$={M_inf})')
    plt.plot(theta_surface, Cp_exact , 'r-' , linewidth=2, label=f'Exact(incompresible) ($M_\infty$={M_inf})')
    plt.plot(theta_surface, Cp_prandtl_glaurt , 'g-' , linewidth=2, label=f'prandtl-glaurt correction ($M_\infty$={M_inf})')
    plt.xlabel(r'$\theta$ (rad)')
    plt.ylabel(r'$C_p$')
    plt.title('Pressure Coefficient on Cylinder Surface')
    plt.grid(True)
    plt.legend()
    plt.show()
    

# plotting u_surface via tetha


theta_surface = np.linspace(0, np.pi, N_tetha)

u_normalized = u_surface / u_inf
u_normalized_exact = -2*np.sin(theta_surface)

plt.figure(figsize=(8, 5))
plt.plot(theta_surface, u_normalized, 'g-', linewidth=2, label=f'Code ($M_\infty$={M_inf})')
plt.plot(theta_surface, u_normalized_exact, 'r-', linewidth=2, label=f'Exact(incompresible) ($M_\infty$={M_inf})')
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$u_\theta / u_\infty$')
plt.title('dimminsionless Velocity on Cylinder Surface')
plt.grid(True)
plt.legend()
plt.show()




theta_surface = np.linspace(0, np.pi, N_tetha)

plt.figure(figsize=(8, 5))
plt.plot(theta_surface, M_surrface, 'b-', linewidth=2, label=f'Surface Mach ($M_\infty$={M_inf})')
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'$M_s$')
plt.title('Mach Number on Cylinder Surface')
plt.grid(True)
plt.legend()
plt.show()


try:
    import time
    while plt.get_fignums():
        plt.pause(2)  
        remaining = len(plt.get_fignums())
finally:
    plt.close('all')
