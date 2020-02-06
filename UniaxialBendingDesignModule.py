"""
Uni-axial Bending Element
========================

This is python module for analysis and design of Uni-axial bending elements
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

FORCE_TOL = 0.001 # Newtons

class Steel:
	"""
	Class for steel
	"""
	def __init__(self,fy=500E6,FOS=1.15, Es=2.1E11):
		self.fy = fy
		self.FOS = FOS
		self.Es = Es
		self.max_tensile_strain = fy/(Es*FOS) + 0.002
		self.max_compresive_strain = -fy/(Es*FOS)
		
	def __call__(self,strain):
		if abs(strain*self.Es) <= (self.fy/self.FOS):
			return strain*self.Es
		else:
			return np.sign(strain)*(self.fy/self.FOS)
	def test(self):
		fID = open('steel_test.csv','w')
		fmt = '{0:8.5f} , {1:15.5f}, Pa\n'
		strain = np.linspace(-0.0035,0.0035,101)
		for str in strain:
			fID.write(fmt.format(str,self(str)))
		fID.close()
	
class BISConcrete:
	"""
	Class for Concrete
	"""
	def __init__(self,fck=30E6, FOS = 1.5):
		self.fck = fck
		self.FOS = FOS
		self.max_tensile_strain = 0.0
		self.max_compresive_strain = -0.0035
		
	def __call__(self, strain):
		if strain >= 0.0:
			return 0.0
		elif strain >= -0.002:
			strain_c = 2*(abs(strain)/0.002) - (abs(strain)/0.002)**2
			return -1.0*0.67*strain_c*self.fck/self.FOS
		elif strain >= -0.0035:
			return -1.0*0.67*self.fck/self.FOS
		else:
			return 0.0
	def test(self):
		fID = open('concrete_test.csv','w')
		fmt = '{0:8.5f} , {1:15.5f}, Pa\n'
		strain = np.linspace(-0.005,0.005,101)
		for str in strain:
			fID.write(fmt.format(str,self(str)))
		fID.close()

class UniaxialBendingSection:
	"""
	Class for Uni-axial Bending
	"""
	def __init__(self,inpfile,meshsize=1001):
		self.inpfile = inpfile
		self.MomentCapacity = {}
		self.read_inp(inpfile, meshsize)
		self.estimate_centroid()
	def estimate_maxm_neutral_axis_depth(self):
		"""
		Class method to estimate maximum neutral axis depth for balanced failure
		for the given sectional geometry.
		"""
		xu_d = abs(self.concrete.max_compresive_strain)/\
			(abs(self.concrete.max_compresive_strain) + self.steel.max_tensile_strain)
		self.max_positive_na = xu_d*self.positive_effective_depth
		self.max_negative_na = xu_d*self.negative_effective_depth

	def estimate_centroid(self):
		"""
		Class method for estimation of centroid of section for estimation of --
		"""
		strain = self.strain_distribution_compr(-0.002,-0.002)
		self.centroid = (self.depth/2)+\
			(self.sectional_moment(strain, self.depth/2)/\
			self.sectional_force(strain))

	def estimate_balance_moment(self):
		# Positive balance section
		self.balance_section = {\
			'POSITIVE':{'Pu':0.0,'Mu':0.0},\
			'NEGATIVE':{'Pu':0.0,'Mu':0.0}}
		strain = self.strain_distribution_capacity(self.max_positive_na,True)

	def strain_distribution(self,na_z,phi):
		"""
		A class method for finding strain distribution for a give na_z and phi

		:para na_z: length to neutral axis from the bottom of section
		:type na_z: float
		:para phi: Curvature of the section. Please see the doc for sign convention
		:type phi: float
		"""
		return (self.mesh_center - na_z)*phi
	def strain_distribution_capacity(self, depth, positive):
		"""
		A class method for finding strain distribution for a given depth from compressive
		fiber and curvature angle.

		:para depth: Depth from top compressive fiber
		:type depth: float
		:para positive: A boolean indicating whether curvature angle is positive or negative
		:type positive: boolean 
		"""
		if positive:
			phi = -self.concrete.max_compresive_strain/depth
			na_z = depth
		else:
			phi = self.concrete.max_compresive_strain/depth
			na_z = self.depth - depth
		return self.strain_distribution(na_z,phi)

	def concrete_stress(self,strain):
		"""
		A class method for estimating stress distribution in concrete of the section

		:para strain: 
		:type strain:
		:return : stress in concrete of the section
		:rtype : numpy.array 
		"""
		stress = np.zeros(len(strain))
		for i,strn in enumerate(strain):
			stress[i] = self.concrete(strn)
		return stress
	def steel_stress(self,strain_dis):
		"""
		A class method for estimating stress distribution in reinforcing bar in the section

		:param strain:
		:type strain:
		:return: Stress field in reinforcing bar
		:rtype: numpy.array
		"""
		stress = np.zeros(len(self.reinforcement))
		for i,steel in enumerate(self.reinforcement):
			strain = np.interp(steel[0], self.mesh_center,strain_dis)
			stress[i] = (self.steel(strain)-self.concrete(strain))
		return stress
	def concrete_total_force(self,strain):
		"""
		Class method for estimating total force due to concrete section.

		:param strain:
		:type strain: numpy.array
		:return: Total force due to concrete section
		:rtype: float
		"""
		stress = self.concrete_stress(strain)
		return sum(stress*(self.width*self.mesh_dz))
	def concrete_total_moment(self,strain,na_z):
		"""
		Class method for estimating total moment due to concrete section.

		:param strain:
		:type strain: numpy.array
		:param na_z: depth to neutral axis from bottom of section
		:type na_z: float
		:return: Total moment due to concrete section
		:rtype: float
		"""
		force = self.concrete_stress(strain)*(self.width*self.mesh_dz)
		return sum(force*(self.mesh_center - na_z))
	def steel_total_force(self,strain_dis):
		"""
		Class method for estimating total force of the section due to reinforcing bars 
		for a given strain distribution.

		:param strain_dis:
		:type strain_dis:
		:return: Total force due to reinforcing bar
		:rtype: float
		"""
		total_force = 0.0
		for steel in self.reinforcement:
			strain = np.interp(steel[0], self.mesh_center,strain_dis)
			force = (self.steel(strain)-self.concrete(strain))*steel[1]
			total_force = total_force + force
		return total_force
	def steel_total_moment(self,strain_dis,na_z):
		"""
		Class method for estimating total moment of the section due to reinforcing vars
		for a given strain distribution around na_z
		"""
		total_moment = 0.0
		for steel in self.reinforcement:
			strain = np.interp(steel[0], self.mesh_center,strain_dis)
			force = (self.steel(strain)-self.concrete(strain))*steel[1]
			total_moment = total_moment + force*(steel[0]-na_z)
		return total_moment
	def section_linearization(self,strain):
		"""
		Linearizion of stress distribution. As per ASME ??????
		"""
		centroidal_axis = self.depth*0.5
		return self.sectional_force(strain),\
			self.sectional_moment(strain,centroidal_axis)
	def sectional_force(self,strain):
		"""
		Class method for estimation of total force of section for a given strain distribution.
		"""
		return self.steel_total_force(strain) + \
			self.concrete_total_force(strain)
	def sectional_moment(self,strain,na_z):
		"""
		Class method for estimation of total moment of section about na_z for a given strain distribution.
		"""
		return self.steel_total_moment(strain,na_z) + \
			self.concrete_total_moment(strain,na_z)

	def compressive_strain(self,Pu,max_iter=100):
		ub = self.steel.max_tensile_strain
		lb = -0.002
		str_c = 0.5*(ub+lb)
		section_converged = False
		for i in range(max_iter):
			strain = self.mesh_center*0.0 + str_c
			Fc = self.concrete_total_force(strain)
			Fs = self.steel_total_force(strain)
			if abs(Fc + Fs - Pu) <= FORCE_TOL:
				section_converged = True
				break
			else:
				if (Fc + Fs - Pu) > 0.0:
					ub = str_c
					str_c = 0.5*(lb+str_c)
				else:
					lb = str_c
					str_c = 0.5*(lb+str_c)
		if section_converged:
			return str_c
		else:
			return None

	def balance_section_capacity(self,postive=True,max_iter=1000):
		"""
		Class method for finding balance section for for a given Capacity
		"""
		ub = self.depth
		lb = 0.0
		na_z = 0.5*self.depth
		section_converged = False
		for i in range(max_iter):
			if postive:
				strain = self.strain_distribution_capacity(na_z,postive)
			else:
				strain = self.strain_distribution_capacity(self.depth-na_z,postive)
			Fc = self.concrete_total_force(strain)
			Fs = self.steel_total_force(strain)
			if abs(Fc + Fs) <= FORCE_TOL:
				section_converged = True
				break
			else:
				if postive:
					if (Fc + Fs) > 0.0:
						lb = na_z
						na_z = 0.5*(na_z + ub)
					else:
						ub = na_z
						na_z = 0.5*(na_z + lb)
				else:
					if (Fc + Fs) > 0.0:
						ub = na_z
						na_z = 0.5*(na_z + lb) 
					else:
						lb = na_z
						na_z = 0.5*(na_z + ub)
		if section_converged:
			return na_z
		else:
			return None

	def balance_section(self,phi, max_iter=100):
		"""
		Class method for finding balance section for a given sectional curvature 
		"""
		ub = self.depth
		lb = 0.0
		na_z = 0.5*self.depth
		section_converged = False
		for i in range(max_iter):
			strain = self.strain_distribution(na_z,phi)
			Fc = self.concrete_total_force(strain)
			Fs = self.steel_total_force(strain)
			if abs(Fc + Fs) <= FORCE_TOL:
				section_converged = True
				break
			else:
				if phi >= 0.0:
					if (Fc + Fs) > 0.0:
						lb = na_z # Non mutable variable fingures crossed.. Found a bug.. Bitch is here
						na_z = 0.5*(na_z + ub)
					else:
						ub = na_z
						na_z = 0.5*(na_z + lb)
				else:
					if (Fc + Fs) > 0.0:
						ub = na_z
						na_z = 0.5*(na_z + lb) 
					else:
						lb = na_z
						na_z = 0.5*(na_z + ub)
		if section_converged:
			return na_z
		else:
			return None
	def read_inp(self,inpfile, meshsize=10000001):
		"""
		Class method for reading an input file.
		"""
		with open(inpfile,'r') as fID:
			data = fID.readlines()
		temp = data[0].strip().split(',')
		self.width = float(temp[0])
		self.depth = float(temp[1])
		self.reinforcement = []
		self.concrete = None
		self.steel = None
		self.mesh = np.linspace(0,self.depth,meshsize)
		self.mesh_dz = abs(self.mesh[0]-self.mesh[1])
		self.mesh_center = self.mesh[:-1] + 0.5*self.mesh_dz
		self.positive_effective_depth = 0.0
		self.negative_effective_depth = 0.0
		for line in data[1:]:
			temp = line.strip().split(',')
			if temp[0] == 'T':
				rein_force_no = [int(bar) for bar in temp[2::2]]
				rein_force_area = [0.25*np.pi*float(bar)*float(bar) for bar in temp[3::2]]
				self.negative_effective_depth = max([self.negative_effective_depth,\
					self.depth - float(temp[1])])
				self.reinforcement.append([self.depth - float(temp[1]),\
					sum(np.array(rein_force_no)*np.array(rein_force_area))])
			elif temp[0] == 'B':
				rein_force_no = [int(bar) for bar in temp[2::2]]
				rein_force_area = [0.25*np.pi*float(bar)*float(bar) for bar in temp[3::2]]
				self.positive_effective_depth = max([self.positive_effective_depth,\
					self.depth - float(temp[1])])
				self.reinforcement.append([float(temp[1]),\
					sum(np.array(rein_force_no)*np.array(rein_force_area))])
			elif temp[0] == 'CONCRETE':
				self.concrete = BISConcrete(float(temp[1]),float(temp[2]))
			elif temp[0] == 'STEEL':
				self.steel = Steel(float(temp[1]),float(temp[2]))
			else:
				print('Unknown input format..')
		'''
		self.depth = depth
		self.width = width
		self.top_layer_reinforcement = top_layer_reinforcement
		self.bottom_layer_reinforcement = bottom_layer_reinforcement
		
		'''
	def moment_capacity(self, postive=True):
		if not(self.MomentCapacity):
			self.estimate_moment_capacity()
		else:
			return (self.MomentCapacity['POSITIVE']['Capacity'] if\
				postive else self.MomentCapacity['NEGATIVE']['Capacity'])
	def estimate_moment_capacity(self):
		na_zp = self.balance_section_capacity(postive=True)
		phi_p = 0.0035/na_zp
		Est_p = phi_p * (self.depth-na_zp)
		strain = self.strain_distribution_capacity(na_zp,True)
		Mp = self.sectional_moment(strain,na_zp)
		na_zn = self.balance_section_capacity(postive=False)
		phi_n = -0.0035/(self.depth-na_zn)
		Est_n = abs(phi_n * na_zn)
		strain = self.strain_distribution_capacity(self.depth-na_zn,False)
		Mn = self.sectional_moment(strain,na_zn)
		self.MomentCapacity = {\
			'POSITIVE':{'Neutral Axis':na_zp,'Capacity':Mp,\
			'PHI':phi_p,'Est':Est_p},\
			'NEGATIVE':{'Neutral Axis':na_zn,'Capacity':Mn,\
			'PHI':phi_n,'Est':Est_n}}
		"""
		self.balance_section_capacity(postive=True)
		limit_phi = (abs(self.concrete.max_compresive_strain)+\
			abs(self.steel.max_tensile_strain))/self.depth
		na_z = self.balance_section(limit_phi)
		print(na_z)
		strain = self.strain_distribution(na_z,limit_phi)
		Mp = self.sectional_moment(strain,na_z)
		na_z = self.balance_section(-1.0*limit_phi)
		strain = self.strain_distribution(na_z,-1.0*limit_phi)
		Mn = self.sectional_moment(strain,na_z)
		return Mp,Mn
		"""
	def minimum_eccentricty(self):
		self.min_eccentricty = min(0.02,\
			self.unsupported_length/500+self.depth/30)

	def axial_capacity(self):
		tensile_strain = self.mesh_center*0.0 + self.steel.max_tensile_strain
		compressive_strain = self.mesh_center*0.0 + -0.002
		return self.sectional_force(tensile_strain),\
			self.sectional_force(compressive_strain)
	def estimate_moment_curvature(self, inter_pnts=51):
		if not(self.MomentCapacity):
			self.estimate_moment_capacity()
		PHI_set = np.linspace(\
			self.MomentCapacity['NEGATIVE']['PHI'],\
			self.MomentCapacity['POSITIVE']['PHI'],inter_pnts)
		MC = np.zeros(len(PHI_set))
		for i,phi in enumerate(PHI_set):
			na_z = self.balance_section(phi)
			strain = self.strain_distribution(na_z,phi)
			MC[i] = self.sectional_moment(strain,na_z)
		self.moment_curvature = {'PHI':PHI_set, 'Mp':MC}

	def strain_distribution_compr(self,strain1,strain2):
		return strain1+self.mesh_center*((strain2-strain1)/self.depth)

	def compressive_strain_given_tension(self,Ec):
		if Ec <= 0.0:
			return self.concrete.max_compresive_strain - 0.75*Ec
		else:
			return self.concrete.max_compresive_strain

	def create_interaction_figure(self,fname='sample2.pdf',scale=0.001,):
		f,ax=plt.subplots(1,1,figsize=(11.7,8.27))
		ax.plot(self.axial_moment_interaction['Mu_p']*scale,\
			self.axial_moment_interaction['Pu_p']*scale)
		ax.plot(self.axial_moment_interaction['Mu_n']*scale,\
			self.axial_moment_interaction['Pu_n']*scale)
		ax.grid(b=True, which='major', color='b', linestyle='--')
		ax.axhline(y=0, color='k',lw=0.25)
		ax.axvline(x=0, color='k',lw=0.25)
		ax2 = inset_axes(ax,width="20%",height="20%",loc=1)
		ax3 = inset_axes(ax,width="20%",height="20%",loc=2)
		self.create_section_figure(ax2)
		self.create_moment_curvature_figure(ax3)
		ax.set_xlabel('Sectional Moment (kNm)')
		ax.set_ylabel('Axial Force (kN)')
		plt.savefig(fname,bbox_inches = 'tight', pad_inches = 0.1)

	def create_moment_curvature_figure(self,ax,scale=0.001):
		ax.yaxis.set_ticks_position('right')
		ax.grid(b=True, which='major', color='b', linestyle='--')
		ax.plot(self.moment_curvature['PHI'],self.moment_curvature['Mp']*scale)
		ax.axhline(y=0, color='k',lw=0.25)
		ax.axvline(x=0, color='k',lw=0.25)
		ax.set_xlabel(r'$\phi$')
		ax.set_ylabel(r'$M_u (kN)$')

	def create_section_figure(self,ax):
		p = patches.Rectangle((0.0, 0.0),\
			self.width, self.depth,\
			facecolor='gray', clip_on=False,alpha=0.5)
		ax.add_patch(p)
		ax.set_xlim([-0.1,self.width+0.1])
		ax.set_ylim([-0.1,self.depth+0.1])
		with open(self.inpfile,'r') as fID:
			data = fID.readlines()
		for line in data:
			temp = line.strip().split(',')
			if (temp[0] == 'T') or (temp[0] == 'B'):
				rein_force_no = sum([int(bar) for bar in temp[2::2]])
				x = np.linspace(0.03, self.width-0.03,rein_force_no)
				if temp[0] == 'T':
					y = x*0.0 + (self.depth - float(temp[1]))
				else:
					y = x*0.0 + float(temp[1])
				ax.scatter(x,y,c='k')
		ax.axhline(y=self.centroid, c='r', lw=0.25)    

	def gen_axial_moment_inter(self,discrete_pnts=1001):
		"""
		Python class method for generating moment interaction curve.
		"""
		Est_p = self.MomentCapacity['POSITIVE']['Est']
		Est_n = self.MomentCapacity['NEGATIVE']['Est']
		Est_positive_set = np.linspace(-0.002,Est_p)
		Est_negative_set = np.linspace(-0.002,Est_n)
		Pu_p = []
		Mu_p = []
		Pu_n = []
		Mu_n = []
		for est in Est_positive_set:
			Es = self.compressive_strain_given_tension(est)
			strains = self.strain_distribution_compr(Es,est)
			Pu_p.append(self.sectional_force(strains))
			Mu_p.append(self.sectional_moment(strains,self.centroid))
		strains = self.strain_distribution_compr(self.steel.max_tensile_strain,\
				self.steel.max_tensile_strain)
		Pu_p.append(self.sectional_force(strains))
		Mu_p.append(self.sectional_moment(strains,self.centroid))
		for est in Est_negative_set:
			Es = self.compressive_strain_given_tension(est)
			strains = self.strain_distribution_compr(est,Es)
			Pu_n.append(self.sectional_force(strains))
			Mu_n.append(self.sectional_moment(strains,self.centroid))
		strains = self.strain_distribution_compr(self.steel.max_tensile_strain,\
				self.steel.max_tensile_strain)
		Pu_n.append(self.sectional_force(strains))
		Mu_n.append(self.sectional_moment(strains,self.centroid))
		self.axial_moment_interaction = {\
			'Pu_p':np.array(Pu_p),'Mu_p':np.array(Mu_p),\
			'Pu_n':np.array(Pu_n),'Mu_n':np.array(Mu_n)}

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	steel = Steel()
	steel.test()
	cc = BISConcrete()
	cc.test()
	section1 = UniaxialBendingSection('SampleSection.dat')
	max_strain = 2*0.0035/section1.depth
	PHI_set = np.linspace(-3.0*max_strain,3*max_strain,51)
	MC = np.zeros(len(PHI_set))
	fig, ax = plt.subplots(1,3,figsize=(30,10))
	for i,phi in enumerate(PHI_set):
		lbl = '{0:8.5f}'.format(phi)
		na_z = section1.balance_section(phi)
		strain = section1.strain_distribution(na_z,phi)
		stress = section1.concrete_stress(strain)
		MC[i] = section1.sectional_moment(strain,na_z)
		ax[0].plot(strain,section1.mesh_center, label=lbl)
		ax[1].plot(stress/10**6,section1.mesh_center, label=lbl)
	ax[2].plot(PHI_set,MC/10**6)
	plt.savefig('stress_strain_validation.pdf')
	limit_phi = (abs(section1.concrete.max_compresive_strain)+\
			abs(section1.steel.max_tensile_strain))/section1.depth
	na_z = section1.balance_section(limit_phi)
	print(na_z,limit_phi)
	strain = section1.strain_distribution(na_z,max_strain)
	stress = section1.concrete_stress(strain)
	OO = np.zeros([len(strain),4])
	OO[:,0] = strain
	OO[:,1] = stress
	OO[:,2] = section1.mesh_center
	OO[:,3] = na_z
#	print(section1.steel_total_force(strain))
#	print(section1.concrete_total_force(strain))
#	print(section1.concrete_total_moment(strain,na_z))
	print(section1.steel.max_tensile_strain)
	print(section1.sectional_moment(strain,na_z))
	print(section1.axial_capacity())
	np.savetxt("strains.csv", OO, delimiter=",")