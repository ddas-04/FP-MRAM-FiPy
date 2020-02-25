
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
    uAxisUnit = uAxis / uAxisNorm
    #################################################
    mArray = mUnit
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

def H_DemagField(mUnit, uAxis, Msat):
	uAxisNorm = numerix.linalg.norm(uAxis)
	uAxisUnit = uAxis / uAxisNorm
        #################################################
        mArray = mUnit
        
	uAxisArr = numerix.tile(uAxisUnit, (len(mUnit[0]), 1)) #converted to nX3
    	uAxisArr = numerix.transpose(uAxisArr) # transposed back to 3Xn 
    	mdotu = numerix.dot(mArray, uAxisArr)  # projection of m along uniaxial direction
    	scaleFac = numerix.multiply(mdotu, (-4*numerix.pi* Msat)) # calculate the magnitude in Tesla
	H_demag = numerix.zeros((3, len(scaleFac)), 'd') # Uniaxial vector for each cell
    	H_demag[0] = numerix.multiply(scaleFac, uAxisArr[0])
    	H_demag[1] = numerix.multiply(scaleFac, uAxisArr[1])
    	H_demag[2] = numerix.multiply(scaleFac, uAxisArr[2])
    	
	return 0*H_demag

# Real time LLG
def Calculate_dmdt(mAllCell,HeffBase):
    global alphaDamping, gamFac, mu0
    H=HeffBase    
    m=mAllCell
    mXH=numerix.cross(m,H,axisa=0,axisb=0)
    precisionTerm=numerix.transpose(mXH)
    
    mXmXH=numerix.cross(m,precisionTerm,axisa=0,axisb=0)
    dampingTerm=(alphaDamping)*numerix.transpose(mXmXH)
    constant_factor=-(gamFac*mu0)
    dmdt=(constant_factor)*(precisionTerm+dampingTerm) 
        
    return dmdt

def normalize_cell(vectorCellArray):
    mag=numerix.linalg.norm(vectorCellArray,axis=0)
    normalized_vector_array=vectorCellArray/mag
    return normalized_vector_array
######################################################################################

# ### Define Mesh

print('Meshing starts')

mesh = Gmsh2DIn3DSpace('''
    radius = 1.0;
    cellSize = 0.05;
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
#print('Gridcoor')

mUnit = gridCoor

mNorm = numerix.linalg.norm(mUnit,axis=0)   

print('max mNorm='+str(max(mNorm))) 
print('min mNorm='+str(min(mNorm))) 
mAllCell = mUnit / mNorm                      # m values around the sphere surface are normalized
#print(mAllCell)

######## Constant terms ################
kBoltzmann = 1.38064852e-23    #in J/K
mu0 = numerix.pi * 4.0e-7      #in N/A^2


############################### LLG parameters ###################################

#gamFac = 1.76086e11              # in rad/(s.T)
gamFac = 2.211e5                     # in 1/(s.T)
alphaDamping = 0.01
Temperature = 300              # in K
Msat =450e3                  # in A/m
#thickness=2e-9
#length=50e-9
#width=3*length
magVolume = 2.0e-9 * (50e-9 * 50e-9) * numerix.pi   # in m^3
#magVolume=(numerix.pi/4)*length*width*thickness
D = alphaDamping * gamFac * kBoltzmann * Temperature / ((1 + alphaDamping**2) *Msat * magVolume) # unit 1/s

print('D = '+str(D))
Eb=44*kBoltzmann*Temperature
Ku2=Eb/magVolume
#print('Ku2 = '+str(Ku2))
#exit()
#Ku2 = 224e3                   # in J/m^3

#D=D*numerix.sqrt(time_scale_fac)
#################################################################################

######## Calculate uniaxial anisotropy field $H_{ani}$ ##################################
uAxis = numerix.array([[0., 0., 1.]])
HuniaxBase = H_UniaxialAnisotropy(mAllCell, uAxis, Ku2, Msat)
#exit()
################################################################################
################ Calculate demag field $H_{demag}$ ##################################
demag_axis=numerix.array([[0., 1., 0.]])
HdemagBase=H_DemagField(mAllCell, demag_axis, Msat)
###############################################################################

HeffBase = HuniaxBase +  HdemagBase    # Effective Field

############################## Time section ######################################
dexp=-25
limit=0.1
incr=0.05
number_of_steps=int((limit-dexp)/incr)+1
#timefac=(gamFac*Hmag)/(1+alphaDamping**2) # unit 1/s
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
Dval=CellVariable(mesh=mesh, value=D)
#print(dmdt)
while dexp<limit-incr:

    print('*************************************************************')
    percentage=float(float(t_i)/(float(number_of_steps)-1.0))*100.0
    print('Completed = '+str(percentage)+'%')

    print('dexp='+str(dexp))
    #timestep=min(1e-3, numerix.exp(dexp))
    timestep=numerix.exp(dexp)
    print('timestep='+str(timestep))
    
    eqI = (TransientTerm()== DiffusionTerm(coeff=Dval)- ExponentialConvectionTerm(coeff=dmdt))
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
    


