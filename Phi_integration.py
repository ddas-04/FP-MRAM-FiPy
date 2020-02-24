#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().magic(u'reset')


# In[167]:


# Phi value read function
def get_phi_tuple():
    Filename = 'with_axis_0_img_00801.vtk' #This file contains the phi value
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(Filename)
    reader.ReadAllScalarsOn()
    reader.Update()
    usg = reader.GetOutput().GetCellData().GetScalars()
    z = []
    for i in range(280918):
        z.append(usg.GetTuple(i)[0])
    return tuple(z)


# In[168]:


# function to convert cartesian to spherical polar co-ordinate
def cart2pol(mValue):
    r=numerix.linalg.norm(mValue,axis=0)  
    theta=np.arccos(mValue[2,:]/r)
    phi=np.arctan2(mValue[1,:],mValue[0,:])
    # this section converts negative phi value to positive value by adding 2pi value
    neg_index=np.where(phi<0)
    phi[neg_index]=phi[neg_index]+2*np.pi
    
    return r,theta,phi
    


# In[10]:


# Importing Modules
#from fipy import FaceVariable, CellVariable, Gmsh2DIn3DSpace, VTKViewer, TransientTerm, ExplicitDiffusionTerm, DiffusionTerm, ExponentialConvectionTerm, DefaultSolver
from fipy import CellVariable
from fipy.variables.variable import Variable
from fipy.tools import numerix
#import time
import pickle
#from shutil import copyfile 
import vtk
import numpy as np


# In[3]:


mesh = pickle.load(open("mesh_details_rad_1_cellsize_0pt01.p","rb"))


# In[4]:


gridCoor = mesh.cellCenters
mUnit = gridCoor
mNorm = numerix.linalg.norm(mUnit,axis=0)   
print('max mNorm='+str(max(mNorm))) 
print('min mNorm='+str(min(mNorm))) 
mAllCell = mUnit / mNorm


# In[7]:


phi_value = get_phi_tuple() # get phi values from a previous .vtk file
phi = CellVariable(name=r"$\Phi$",mesh=mesh,value=phi_value)


# In[13]:


# save phi values in numpy array file
np.save('phi_value.npy', phi)


# In[14]:


phi_value=[]


# In[21]:


#load numpy array of phi values
phi_value=np.load('phi_value.npy')


# In[23]:


type(phi_value)


# In[25]:


#save coordinates of the state space into a numpy array
np.save('mValue_info.npy', mAllCell) 


# In[26]:


# load the state space coordinate array values
mValue=np.load('mValue_info.npy')


# In[152]:


np.shape(mValue)


# In[ ]:





# In[180]:


mvalue_sph_pol=np.asarray(cart2pol(mValue))


# In[181]:


type(mvalue_sph_pol)


# In[182]:


print(mvalue_sph_pol)


# In[ ]:





# In[ ]:




