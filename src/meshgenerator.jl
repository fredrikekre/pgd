abstract mesh
abstract spaceMesh <: mesh

type mesh1D <: spaceMesh
    ex::Array{Float64,2} # element coordinates
    x::Array{Float64,1} # coordinates
    edof::Array{Int64,2} # element dofs
    nEl::Int64 # number of elements
    nDofs::Int64 # number of dofs
    nElDofs::Int64 # number of dofs per element
    nElNodes::Int64 # number of nodes per element (2 for linear, 3 for quadratic)
    uniform::Bool # uniform mesh? If true: Speeds up integration
end

type mesh2D <: spaceMesh
    ex::Array{Float64,2}
    ey::Array{Float64,2}
    mesh::Array{Int64,2}
    edof::Array{Int64,2}
    b1::Array{Int64,2}
    b2::Array{Int64,2}
    b3::Array{Int64,2}
    b4::Array{Int64,2}
    nEl::Int64
    nElx::Int64
    nEly::Int64
    nDofs::Int64
    nElDofs::Int64
    coord::Array{Float64,2}
    uniform::Bool
end

function create_mesh1D(sCoord,eCoord,nEl,nElNodes,nNodeDofs)

    nNodes = (nElNodes-1)*nEl + 1
    nDofs = nNodes*nNodeDofs
    edof = zeros(Int64,nElNodes*nNodeDofs,nEl)

    x = linspace(sCoord,eCoord,nNodes)

    if nElNodes == 2
        ex = [x[1:end-1] x[2:end]]'
        mesh = [1:nNodes-1 2:nNodes]'
    elseif nElNodes == 3
        ex = [x[1:2:end-2] x[2:2:end-1] x[3:2:end]]'
        mesh = [1:2:nNodes-2 2:2:nNodes-1 3:2:nNodes]'
    end

    for i = 1:nNodeDofs
        edof[i:nNodeDofs:((nElNodes-1)*nNodeDofs+i),:] = mesh*nNodeDofs-(nNodeDofs-i)
    end
    
    return mesh1D(ex,x,edof,nEl,nDofs,nElNodes*nNodeDofs,nElNodes,true)
end

"""
mesh2D(xlength,ylength,nElx,nEly,nNodeDofs) -> twoDimensionalMesh

Generates a rectangular mesh with rectangular elements

Input:
    xlength/ylength - Length of domain in x/y-direction
    nElx/nEly - Number of elements in x/y-direction
    nNodeDofs - Number of degrees of freedom per node

Output:
    QuadraticMesh(ex,ey,mesh,edof,b1,b2,b3,b4,nEl,nDofs)

    ex/ey - element coordinates
    mesh - element nodes
    edof - topology matrix
    b1-b4 - matrices containing the dofs at the boundaries (see figure)
    nEl - number of elements
    nDofs - number of degrees of freedom
    uniform - true (uniform mesh)

    _________b3________
    |                 |
    |                 |
    |                 |
    b4                b2
    |                 |
    |                 |
    |________b1_______|


"""
function create_mesh2D(xStart,xEnd,yStart,yEnd,nElx,nEly,nNodeDofs)
    nEl = nElx*nEly
    nxNodes = nElx + 1; nyNodes = nEly + 1
    nNodes = nxNodes*nyNodes
    nDofs = nxNodes*nyNodes*nNodeDofs
    nElDofs = 4*nNodeDofs

    ex = zeros(Float64,4,nEl)
    ey = zeros(Float64,4,nEl)
    mesh = zeros(Int,4,nEl)

    X = linspace(xStart,xEnd,nxNodes)
    Y = linspace(yStart,yEnd,nyNodes)
    cEl = 1

    for i = 1:nEly, j = 1:nElx
        ex[:,cEl] = [X[j:j+1]; flipdim(X[j:j+1],1)]
        ey[:,cEl] = [Y[i], Y[i], Y[i+1], Y[i+1]]
        mesh[:,cEl] = [(nxNodes*(i-1)+j):(nxNodes*(i-1)+j+1); flipdim(((nxNodes*(i-1)+j):(nxNodes*(i-1)+j+1))+nxNodes,1)]
        cEl += 1
    end

    cNode = 1
    coord = zeros(2,nNodes)
    for i = 1:nyNodes, j = 1:nxNodes
        coord[:,cNode] = [X[j],Y[i]]
        cNode += 1
    end

    # Make edof and boundaries
    edof = zeros(Int64,4*nNodeDofs,nEl)
    b1 = zeros(Int64,nNodeDofs,nxNodes)
    b2 = zeros(Int64,nNodeDofs,nyNodes)
    b3 = zeros(Int64,nNodeDofs,nxNodes)
    b4 = zeros(Int64,nNodeDofs,nyNodes)

    for i = 1:nNodeDofs
        edof[i:nNodeDofs:(3*nNodeDofs+i),:] = mesh*(nNodeDofs)-(nNodeDofs-i)
        b1[i,:] = (1:nxNodes)*nNodeDofs - (nNodeDofs-i)
        b2[i,:] = (nxNodes:nxNodes:nNodes)*nNodeDofs - (nNodeDofs-i)
        b3[i,:] = ((nNodes-nxNodes+1):nNodes)*nNodeDofs - (nNodeDofs-i)
        b4[i,:] = (1:nxNodes:(nNodes-nxNodes+1))*nNodeDofs - (nNodeDofs-i)
    end

    return mesh2D(ex,ey,mesh,edof,b1,b2,b3,b4,nEl,nElx,nEly,nDofs,nElDofs,coord,true)
end

