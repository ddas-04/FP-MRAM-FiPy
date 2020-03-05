#!/usr/bin/env python
# coding: utf-8

# In[402]:


#get_ipython().magic(u'reset')


# ### Phi value read function

# In[403]:


def get_phi_tuple(number_of_cells):
    Filename = './with_axis_0_VTK_files/with_axis_0_img_00000.vtk' #This file contains the phi value
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(Filename)
    reader.ReadAllScalarsOn()
    reader.Update()
    usg = reader.GetOutput().GetCellData().GetScalars()
    z = []
    for i in range(number_of_cells):
        z.append(usg.GetTuple(i)[0])
    return tuple(z)


# ### function to convert cartesian to spherical polar co-ordinate

# In[404]:


def cart2pol(mValue):
    r=numerix.linalg.norm(mValue,axis=0)  
    theta=np.arccos(mValue[2,:]/r)
    phi=np.arctan2(mValue[1,:],mValue[0,:])
    # this section converts negative phi value to positive value by adding 2pi value
    neg_index=np.where(phi<0)
    phi[neg_index]=phi[neg_index]+2*np.pi
    
    return r,theta,phi


# ### Index sorting function

# In[406]:


def index_sorted(x,y):
    d = np.argsort(x)
    z = [0] * len(y)
    for i in range(len(d)):
        z[i] = y[d[i]]
    return z


# ### Trapezoidal function for irregular grid size

# In[407]:


def trapz_irregular(x,y):
    n=len(x)
    index=1
    sum=0
    while index<=n-1:
        dx=x[index]-x[index-1]
        fx=0.5*(y[index-1]+y[index])
        sum=sum+fx*dx
        index=index+1
    
    return sum


# ### Importing Modules

# In[408]:


print('Import starts')


# In[409]:


from fipy import FaceVariable, CellVariable, Gmsh2DIn3DSpace, VTKViewer, TransientTerm, ExplicitDiffusionTerm, DiffusionTerm, ExponentialConvectionTerm, DefaultSolver
from fipy.variables.variable import Variable
from fipy.tools import numerix
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
from shutil import copyfile 
import vtk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})


# In[410]:


print('Import finished')


# ### Load mesh details from a saved file

# In[411]:


mesh=pickle.load(open("mesh_details_cellsize_0pt05_extrude_1pt00001.p","rb"))
gridCoor = mesh.cellCenters
mUnit = gridCoor
mNorm = numerix.linalg.norm(mUnit,axis=0)   
print('max mNorm='+str(max(mNorm))) 
print('min mNorm='+str(min(mNorm))) 
mAllCell = mUnit / mNorm


# In[412]:


msize=numerix.shape(mAllCell)
number_of_cells=msize[1]
print(number_of_cells)


# ### Loading phi values from a saved file

# In[413]:


phi_value = get_phi_tuple(number_of_cells) # get phi values from a previous .vtk file
phi = CellVariable(name=r"$\Phi$",mesh=mesh,value=phi_value)


# ### To convert into numpy array, save it in numpy file and again store it in a variable

# In[414]:


np.save('phi_value.npy', phi)


# In[415]:


rho_value=[]
rho_value=np.load('phi_value.npy')


# In[416]:


type(rho_value)


# ### Convert m values into numpy array by saving and loading into a file

# In[417]:


np.save('mValue_info.npy', mAllCell) 


# In[418]:


# load the state space coordinate array values
mValue=np.load('mValue_info.npy')
np.shape(mValue)


# ### Convert m values from cartesian to spherical polar

# In[419]:


mvalue_sph_pol=np.asarray(cart2pol(mValue))
type(mvalue_sph_pol)


# In[420]:


phi_angle_all=mvalue_sph_pol[2,:]

phi_value=0
delta=0.02001
phi_save=[]
theta_max=[]
integ_rho_sin_theta=[]
i=0
while phi_value<=(2*np.pi):
    phi_save=np.append(phi_save,phi_value)    # save phi value for 2nd integration
    
    ######### Search index of phi where.....phi_value-delta<=phi_angle_all<phi_value+delta ##############
    index_phi=np.asarray(np.where((phi_angle_all>=phi_value-delta) & (phi_angle_all<phi_value+delta)))
    #####################################################################################################
    size_index_phi=np.shape(index_phi)
    
    theta_at_index=mvalue_sph_pol[1,index_phi] # pick up the theta values
    
    #theta_max=np.append(theta_max,max(theta_at_index)) 
    
    ####################### Sorting rho according to sorted index of theta ########################
    xvar=theta_at_index[0,:] 
    
    rho_at_index=rho_value[index_phi]
    
    yvar=rho_at_index[0,:]
    
    rho_sorted_acc_to_theta=index_sorted(xvar,yvar) # calling the index_sorted function
    rho_sorted_acc_to_theta=np.reshape(rho_sorted_acc_to_theta,(1,len(rho_sorted_acc_to_theta))) # reshaping into a single array
    
    theta_sorted=np.sort(theta_at_index) # theta value sorted
    theta_sorted=theta_sorted[0,:] # converting from 1Xn matrix to row array of n elements 
    
    rho_sin_theta=rho_sorted_acc_to_theta*np.sin(theta_sorted) # calculating rho*sin(theta) part
    rho_sin_theta=rho_sin_theta[0,:] # converting from 1Xn matrix to row array of n elements 
    
    integ_rho_sin_theta_for_phi_val=trapz_irregular(theta_sorted,rho_sin_theta) #integration for irregular grid size
    
    integ_rho_sin_theta=np.append(integ_rho_sin_theta,integ_rho_sin_theta_for_phi_val) 
    
    phi_value=phi_value+delta
        
    

    


# In[421]:


total_Probability=trapz_irregular(phi_save,integ_rho_sin_theta)


# In[422]:


print(total_Probability)


# In[ ]:




