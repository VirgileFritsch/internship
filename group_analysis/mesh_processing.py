import numpy as	 np
import nipy.neurospin.graph.graph as fg

def vectp(a,b):
    """
    vect product of two vectors in 3D
    """
    return np.array([a[1]*b[2]-a[2]*b[1],-a[0]*b[2]+a[2]*b[0],a[0]*b[1]-a[1]*b[0]])

def area (a,b):
    """
    area spanned by the vectors(a,b) in 3D
    """
    c = vectp(a,b)
    return np.sqrt((c**2).sum())

def mesh_to_graph(mesh):
	"""
	This function builds an fff graph from a mesh
	"""
	vertices = np.array(mesh.vertex())
	poly  = mesh.polygon()

	V = len(vertices)
	E = poly.size()
	edges = np.zeros((3*E,2))
	weights = np.zeros(3*E)
	poly  = mesh.polygon()

	for i in range(E):
		sa = poly[i][0]
		sb = poly[i][1]
		sc = poly[i][2]
		
		edges[3*i] = np.array([sa,sb])
		edges[3*i+1] = np.array([sa,sc])
		edges[3*i+2] = np.array([sb,sc])	
			
	G = fg.WeightedGraph(V,edges,weights)

	# symmeterize the graph
	G.symmeterize()

	# remove redundant edges
	G.cut_redundancies()

	# make it a metric graph
	G.set_euclidian(vertices)

	return G

def node_area(vertices, polygons):
    """
    returns a vector of are values, one for each mesh,
    which is the averge area of the triangles around it 
    """
    from numpy.linalg import det
    coord = np.zeros((3,3))
    E = polygons.shape[0]

    narea = np.zeros(len(vertices))
    for i in range(E):
        sa = polygons[i][0]
        sb = polygons[i][1]
        sc = polygons[i][2]
        a = vertices[sa]-vertices[sc]
        b = vertices[sb]-vertices[sc]
        ar = area(a,b)
        narea[sa] += ar
        narea[sb] += ar
        narea[sc] += ar
        
    narea/=6
    # because division by 2 has been 'forgotten' in area computation
    # the area of a triangle is divided into the 3 vertices
    return narea


def mesh_area(mesh):
    """
    This function computes the input mesh area
    """
    vertices = np.array(mesh.vertex())
    poly  = mesh.polygon()
    marea = 0
    coord = np.zeros((3,3))
    E = poly.size()

    for i in range(E):
        sa = poly[i][0]
        sb = poly[i][1]
        sc = poly[i][2]
        a = vertices[sa]-vertices[sc]
        b = vertices[sb]-vertices[sc]
        marea += area(a,b)
    return marea/2

def mesh_integrate(mesh,tex,coord = None):
    """
    Compute the integral of the texture on the mesh
    - coord is an additional set of coordinates to define the vertex position
    by default, mesh.vertex() is used
     """
    from numpy.linalg import det
    if coord == None:
        vertices = np.array(mesh.vertex())
    else:
        vertices = coord
    poly  = mesh.polygon()
    integral = 0
    coord = np.zeros((3,3))
    E = poly.size()
    data = np.array(tex.data())
    def vectp(a,b):
        """
        vect product of two vectors in 3D
        """
        return np.array([a[1]*b[2]-a[2]*b[1],-a[0]*b[2]+a[2]*b[0],a[0]*b[1]-a[1]*b[0]])

    def area (a,b):
        """
        area spanned by the vectors(a,b) in 3D
        """
        c = vectp(a,b)
        return np.sqrt((c**2).sum())
        
    for i in range(E):
        sa = poly[i][0]
        sb = poly[i][1]
        sc = poly[i][2]
        a = vertices[sa]-vertices[sc]
        b = vertices[sb]-vertices[sc]
        mval = (data[sa]+data[sb]+data[sc])/3
        integral += mval*area(a,b)/2
        
    return integral



def flatten(mesh):
    """
    This function flattens the input mesh
    """
    import fff.NLDR
    G = mesh_to_graph(mesh)

    chart = fff.NLDR.isomap_dev(G,dim=2,p=300,verbose = 0)
    
    #print np.shape(chart)
    vertices = np.array(mesh.vertex())
    
    for i in range(G.V):
        mesh.vertex()[i][0]=chart[i,0]
        mesh.vertex()[i][1]=chart[i,1]
        mesh.vertex()[i][2]= 0
        
    mesh.updateNormals()
    
    return mesh

def write_aims_Mesh(vertex, polygon, fileName):
    """
    Given a set of vertices, polygons and a filename,
    write the corresponding aims mesh
    the aims mesh is returned
    """
    from soma import aims
    vv = aims.vector_POINT3DF()
    vp = aims.vector_AimsVector_U32_3()
    for x in vertex: vv.append(x)
    for x in polygon: vp.append(x)
    m = aims.AimsTimeSurface_3()
    m.vertex().assign( vv )
    m.polygon().assign( vp )
    m.updateNormals()
    W = aims.Writer()
    W.write(m, fileName)
    return m

def smooth_texture_from_mesh(mesh, input_texture, output_texture, sigma, lsigma=1.):
   """
   Smooth a texture along some mesh

   parameters
   ----------
   mesh: string,
         path to AIMS mesh
   input_texture: string,
                  AIMS texture path
   ouput_texture: string,
                  AIMS texture path
   sigma: float,
          desired amount of smoothing
   lsigma: float,
           approximate smoothing in one iteration
   """
   import nipy.neurospin.glm_files_layout.tio as tio
   import nipy.neurospin.graph.field as ff
   from soma import aims
   
   R = aims.Reader()
   G = mesh_to_graph(R.read(mesh))
   add_edges = np.vstack((np.arange(G.V), np.arange(G.V))).T
   edges = np.vstack((G.edges, add_edges))
   weights = np.concatenate((G.weights, np.zeros(G.V)))

   f = ff.Field(G.V, edges, weights)   
   f.weights = np.exp(-f.weights**2/(2*lsigma**2))
   f.normalize(0)
   niter = (sigma/lsigma)**2
   
   data = tio.Texture("").read(input_texture).data
   data[np.isnan(data)] = 0
   f.set_field(data.T)
   f.diffusion(niter)
   data = f.get_field()
   
   tio.Texture("", data=data.T).write(output_texture)
   return data

   
