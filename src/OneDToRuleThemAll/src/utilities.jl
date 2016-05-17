function build_function{T}(Ux::PGDComponent, ax::Vector{Vector{T}},
                          Uy::PGDComponent, ay::Vector{Vector{T}},
                          Uz::PGDComponent, az::Vector{Vector{T}})

    length(ax) == length(ay) == length(az) || throw(ArgumentError("Noob."))

    Uxdofs = 1:3:(Ux.mesh.nDofs-2)
    Vxdofs = 2:3:(Ux.mesh.nDofs-1)
    Wxdofs = 3:3:(Ux.mesh.nDofs-0)

    Uydofs = 1:3:(Uy.mesh.nDofs-2)
    Vydofs = 2:3:(Uy.mesh.nDofs-1)
    Wydofs = 3:3:(Uy.mesh.nDofs-0)

    Uzdofs = 1:3:(Uz.mesh.nDofs-2)
    Vzdofs = 2:3:(Uz.mesh.nDofs-1)
    Wzdofs = 3:3:(Uz.mesh.nDofs-0)

    lax = div(length(ax[1]),3)
    lay = div(length(ay[1]),3)
    laz = div(length(az[1]),3)

    U = zeros(lax,lay,laz)
    V = zeros(lax,lay,laz)
    W = zeros(lax,lay,laz)

    number_of_modes = length(ax)

    for mode in 1:number_of_modes
        Ui = vec_mul_vec_mul_vec(ax[mode][Uxdofs],ay[mode][Uydofs],az[mode][Uzdofs])
        U += Ui
        Vi = vec_mul_vec_mul_vec(ax[mode][Vxdofs],ay[mode][Vydofs],az[mode][Vzdofs])
        V += Vi
        Wi = vec_mul_vec_mul_vec(ax[mode][Wxdofs],ay[mode][Wydofs],az[mode][Wzdofs])
        W += Wi
    end

    return U, V, W
end

function vec_mul_vec_mul_vec{T}(x::Vector{T},y::Vector{T},z::Vector{T})
    lx = length(x); ly = length(y); lz = length(z)

    data = T[]
    for k in 1:lz, j in 1:ly, i in 1:lx
        push!(data,x[i]*y[j]*z[k])
    end
    return reshape(data,(lx,ly,lz))
end

type IterativeFunctionComponents{T}
    Ux::Vector{T}
    Uy::Vector{T}
    Uz::Vector{T}
end

function reset(comps::IterativeFunctionComponents)
    lx = length(comps.Ux)
    ly = length(comps.Uy)
    lz = length(comps.Uz)
    return IterativeFunctionComponents(ones(lx), ones(ly), ones(lz))
end

function iteration_difference(compsnew::IterativeFunctionComponents, compsold::IterativeFunctionComponents)
    xdiff = norm(compsnew.Ux - compsold.Ux)/norm(compsold.Ux)
    ydiff = norm(compsnew.Uy - compsold.Uy)/norm(compsold.Uy)
    zdiff = norm(compsnew.Uz - compsold.Uz)/norm(compsold.Uz)

    return xdiff, ydiff, zdiff
end


function extract_eldofs{T}(a::Vector{Vector{T}}, m::Vector{Int})
    b = Vector{T}[]
    for i in 1:length(a)
        push!(b, a[i][m])
    end
    return b
end
