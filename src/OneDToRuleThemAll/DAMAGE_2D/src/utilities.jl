function build_u_function{T}(Ux::PGDComponent, u_ax::Vector{Vector{T}},
                             Uy::PGDComponent, u_ay::Vector{Vector{T}})

    length(u_ax) == length(u_ay) || throw(ArgumentError("Noob."))

    Uxdofs = 1:2:(Ux.mesh.nDofs-1)
    Vxdofs = 2:2:(Ux.mesh.nDofs-0)

    Uydofs = 1:2:(Uy.mesh.nDofs-1)
    Vydofs = 2:2:(Uy.mesh.nDofs-0)

    lax = div(Ux.mesh.nDofs,2)
    lay = div(Uy.mesh.nDofs,2)
    laz = 1

    U = zeros(lax,lay,laz)
    V = zeros(lax,lay,laz)
    W = zeros(lax,lay,laz)

    number_of_modes = length(u_ax)

    for mode in 1:number_of_modes
        Ui = vec_mul_vec_mul_vec(u_ax[mode][Uxdofs],u_ay[mode][Uydofs],[1.0])
        U += Ui
        Vi = vec_mul_vec_mul_vec(u_ax[mode][Vxdofs],u_ay[mode][Vydofs],[1.0])
        V += Vi
    end

    return U, V, W
end
function build_d_function{T}(Dx::PGDComponent, d_ax::Vector{Vector{T}},
                             Dy::PGDComponent, d_ay::Vector{Vector{T}})

    length(d_ax) == length(d_ay) || throw(ArgumentError("Noob."))

    lax = Dx.mesh.nDofs
    lay = Dy.mesh.nDofs
    laz = 1

    D = zeros(lax,lay,laz)

    number_of_modes = length(d_ax)

    for mode in 1:number_of_modes
        Di = vec_mul_vec_mul_vec(d_ax[mode],d_ay[mode],[1.0])
        D += Di
    end

    return D
end

function vec_mul_vec_mul_vec{T}(x::Vector{T},y::Vector{T},z::Vector{T})
    lx = length(x); ly = length(y); lz = length(z)

    data = T[]
    for k in 1:lz, j in 1:ly, i in 1:lx
        push!(data,x[i]*y[j]*z[k])
    end
    return reshape(data,(lx,ly,lz))
end

type IterativeFunctionComponents{Dim,T}
    U::Vector{Vector{T}}
    dims::Val{Dim}
end

# function IterativeFunctionComponents{T}(x::Vector{T},y::Vector{T},z::Vector{T})
#     return IterativeFunctionComponents(Vector{T}[x,y,z],Val{3}())
# end

function IterativeFunctionComponents{T}(x::Vector{T},y::Vector{T})
    return IterativeFunctionComponents(Vector{T}[x,y],Val{2}())
end

# function reset{T}(comps::IterativeFunctionComponents{3,T})
#     lx = length(comps.U[1])
#     ly = length(comps.U[2])
#     lz = length(comps.U[3])
#     return IterativeFunctionComponents(Vector{T}[ones(T,lx), ones(T,ly), ones(T,lz)], Val{3}())
# end

# function reset{T}(comps::IterativeFunctionComponents{2,T})
#     lx = length(comps.U[1])
#     ly = length(comps.U[2])
#     return IterativeFunctionComponents(Vector{T}[ones(T,lx), ones(T,ly)], Val{2}())
# end

# function iteration_difference(compsnew::IterativeFunctionComponents{3}, compsold::IterativeFunctionComponents{3})
#     xdiff = norm(compsnew.U[1] - compsold.U[1])/norm(compsold.U[1])
#     ydiff = norm(compsnew.U[2] - compsold.U[2])/norm(compsold.U[2])
#     zdiff = norm(compsnew.U[3] - compsold.U[3])/norm(compsold.U[3])

#     return xdiff, ydiff, zdiff
# end


function iteration_difference(compsnew::IterativeFunctionComponents{2}, compsold::IterativeFunctionComponents{2})
    xdiff = norm(compsnew.U[1] - compsold.U[1])/norm(compsold.U[1])
    ydiff = norm(compsnew.U[2] - compsold.U[2])/norm(compsold.U[2])

    return xdiff, ydiff
end


function extract_eldofs{T}(a::Vector{Vector{T}}, m::Vector{Int})
    b = Vector{T}[]
    for i in 1:length(a)
        push!(b, a[i][m])
    end
    return b
end
