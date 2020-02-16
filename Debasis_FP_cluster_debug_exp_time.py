
print('Import starts')
from fipy import FaceVariable, CellVariable, Gmsh2DIn3DSpace, VTKViewer, TransientTerm, ExplicitDiffusionTerm, DiffusionTerm, ExponentialConvectionTerm, DefaultSolver
from fipy.variables.variable import Variable
from fipy.tools import numerix
import time
from shutil import copyfile 
print('Import complete')

######################## Function section ###############################

def H_UniaxialAnisotropy(mUnit, uAxis, Ku2, Msat):
    ############ calculate normalized direction #####
    uAxisNorm = numerix.linalg.norm(uAxis)
    #print(uAxisNorm)          
    uAxisUnit = uAxis / uAxisNorm
    #print(uAxisUnit)
    #################################################
    ############ normalizes the grid coordinate #####
    mNorm = numerix.linalg.norm(mUnit,axis=0)     
    mArray = mUnit / mNorm
    #################################################
    #################################################
    # repeat uniaxial direction vector for n times 
    # where n= number of cells. uAxisUnit=3X1
    # uAxisArr=nX3, represents uniaxial direction for
    # each cells in the unit sphere
    #################################################
    uAxisArr = numerix.tile(uAxisUnit, (len(mUnit[0]), 1)) 
    uAxisArr = numerix.transpose(uAxisArr) # converted to 3Xn 
    mdotu = numerix.dot(mArray, uAxisArr)  # projection of m along uniaxial direction
    scaleFac = numerix.multiply(mdotu, (2.0 * Ku2 / Msat)) # calculate the magnitude in Tesla
    Heff = numerix.zeros((3, len(scaleFac)), 'd') # Uniaxial vector for each cell
    Heff[0] = numerix.multiply(scaleFac, uAxisArr[0])
    Heff[1] = numerix.multiply(scaleFac, uAxisArr[1])
    Heff[2] = numerix.multiply(scaleFac, uAxisArr[2])
    return Heff

def Calculate_dmdt(mAllCell,HeffBase):
    global alphaDamping, gamFac, Ku2, Msat
    #Hk=2.0 * Ku2 / Msat
    #Hmag=numerix.linalg.norm(HeffBase,axis=0)
    Hmag=-2.0 * Ku2 / Msat
    timefac=(gamFac*Hmag)/(1+alphaDamping**2) # unit 1/s
    h=HeffBase/Hmag                    # normalized vector along the effective field
    
    m=mAllCell
    mXh=numerix.cross(m,h,axisa=0,axisb=0)
    precisionTerm=(-1)*numerix.transpose(mXh)
    
    mXmXh=numerix.cross(m,precisionTerm,axisa=0,axisb=0)
    dampingTerm=(-alphaDamping)*numerix.transpose(mXmXh)
    
    dmdtau=precisionTerm+dampingTerm # unitless dmdt
    dmdt=timefac*dmdtau
    
    return dmdtau

def normalize_cell(vectorCellArray):
    mag=numerix.linalg.norm(vectorCellArray,axis=0)
    normalized_vector_array=vectorCellArray/mag
    return normalized_vector_array
######################################################################################

# ### Define Mesh

print('Meshing starts')

mesh = Gmsh2DIn3DSpace('''
    radius = 1.0;
    cellSize = 0.01;
    // create inner 1/8 shell
    Point(1) = {0, 0, 0, cellSize};
    Point(2) = {-radius, 0, 0, cellSize};
    Point(3) = {0, radius, 0, cellSize};
    Point(4) = {0, 0, radius, cellSize};
    Circle(1) = {2, 1, 3};
    Circle(2) = {4, 1, 2};
    Circle(3) = {4, 1, 3};
    Line Loop(1) = {1, -3, 2} ;
    Ruled Surface(1) = {1};
    // create remaining 7/8 inner shells
    t1[] = Rotate {{0,0,1},{0,0,0},Pi/2} {Duplicata{Surface{1};}};
    t2[] = Rotate {{0,0,1},{0,0,0},Pi} {Duplicata{Surface{1};}};
    t3[] = Rotate {{0,0,1},{0,0,0},Pi*3/2} {Duplicata{Surface{1};}};
    t4[] = Rotate {{0,1,0},{0,0,0},-Pi/2} {Duplicata{Surface{1};}};
    t5[] = Rotate {{0,0,1},{0,0,0},Pi/2} {Duplicata{Surface{t4[0]};}};
    t6[] = Rotate {{0,0,1},{0,0,0},Pi} {Duplicata{Surface{t4[0]};}};
    t7[] = Rotate {{0,0,1},{0,0,0},Pi*3/2} {Duplicata{Surface{t4[0]};}};
    // create entire inner and outer shell
    Surface Loop(100)={1,t1[0],t2[0],t3[0],t7[0],t4[0],t5[0],t6[0]};
''', order=2).extrude(extrudeFunc=lambda r: 1.01 * r) # doctest: +GMSH

print('Meshing Done')

gridCoor = mesh.cellCenters

mUnit = gridCoor
mNorm = numerix.linalg.norm(mUnit,axis=0)   
print('max mNorm='+str(max(mNorm))) 
print('min mNorm='+str(min(mNorm))) 
mAllCell = mUnit / mNorm


######## Constant terms ################
kBoltzmann = 1.38064852e-23    #in J/K
mu0 = numerix.pi * 4.0e-7      #in N/A^2


############################### LLG parameters ###################################

gamFac = 2.2128e5              # in 1/(s.T)
alphaDamping = 0.01
Temperature = 300              # in K
Msat = 1050e3                  # in A/m
magVolume = 2.0e-9 * (25e-9 * 25e-9) * numerix.pi   # in m^3
D = alphaDamping * gamFac * kBoltzmann * Temperature / ((1 + alphaDamping) * Msat * magVolume) # unit 1/s

Ku2 = 800e3                   # in J/m^3
uAxis = numerix.array([[0., 0., 1.]])
Hmag=2.0 * Ku2 / Msat
#################################################################################

######## Calculate uniaxial anisotropy field $H_{ani}$ ##################################
HuniaxBase = H_UniaxialAnisotropy(mAllCell, uAxis, Ku2, Msat)
################################################################################

HeffBase = HuniaxBase      # Effective Field

############################## Time section ######################################
dexp=-40
limit=0.05
incr=0.05
number_of_steps=int((limit-dexp)/incr)+1
timefac=(gamFac*Hmag)/(1+alphaDamping**2) # unit 1/s
################################################################################

######################## Define Cell variable and Viewer ############################
phi = CellVariable(name=r"$\Phi$",mesh=mesh,value=0.25 / numerix.pi)

viewer=VTKViewer(vars=phi,datamin=0., datamax=1.)
viewer.plot(filename="trial.vtk")
###############################################################################
t_i=0
filename = str(t_i).zfill(5)
dest_name = './with_axis_0_VTK_files/with_axis_0_img_' + str(filename) + '.vtk'  #Path and name of the intermediate file. The Green Part should be changed to your path & name
copyfile('./trial.vtk',dest_name)

dmdt_val=Calculate_dmdt(mAllCell,HeffBase)
dmdt=CellVariable(mesh=mesh, value=dmdt_val)

while dexp<limit-incr:

    print('*************************************************************')
    percentage=float(float(t_i)/(float(number_of_steps)-1.0))*100.0
    print('Completed = '+str(percentage)+'%')

    print('dexp='+str(dexp))
    #timestep=min(1e-3, numerix.exp(dexp))
    timestep=numerix.exp(dexp)
    print('timestep='+str(timestep))
    
    eqI = (TransientTerm()== DiffusionTerm(coeff=D)- ExponentialConvectionTerm(coeff=dmdt))
    
    eqI.solve(var=phi,dt=timestep)
    print('Max phi='+str(max(phi)))
    
    if __name__ == "__main__":#Parameters to be changed are in the seciton below.
        viewer.plot(filename="trial.vtk") #It will only save the final vtk file. You can change the name
    if not t_i == 0:
        filename = str(t_i+1).zfill(5)
        dest_name = './with_axis_0_VTK_files/with_axis_0_img_' + str(filename) + '.vtk'  #Path and name of the intermediate file. The Green Part should be changed to your path & name
        copyfile('./trial.vtk',dest_name) #Specify the path of your trial.vtk file
    dexp=dexp+0.05
    t_i=t_i+1
    


