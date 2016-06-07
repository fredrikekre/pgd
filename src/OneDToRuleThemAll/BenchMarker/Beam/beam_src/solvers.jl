
function u_solver{T,dim,FS,M}(u::Vector{T},u_fe_values::FEValues{dim,T,FS},mesh::FEmesh{dim,T,M},u_free::Vector{Int},
                              E::Tensor{4,dim,T},b::Tensor{1,dim,T})

    n_free_dofs = length(u_free)
    n_dofs = number_of_dofs(mesh)
    Δu = zeros(n_free_dofs)
    u_tri = zeros(n_dofs)

    TOL = 1e-5; i = -1
    while true; i += 1
        copy!(u_tri, u)
        u_tri[u_free] += Δu

        g = zeros(n_dofs)
        _K = start_assemble()

        for ele in 1:mesh.nEl
            m = get_el_dofs(mesh,ele)
            x = get_el_coords(mesh,ele)
            reinit!(u_fe_values,x)

            N, dN = get_shape_functions(u_fe_values)

            ge, Ke = elastic_3D_element(u_tri[m], N, dN, u_fe_values, E, b)

            g[m] += ge
            assemble(m, _K, Ke)
        end

        maxg = maximum(abs(g[u_free]))
        println("maxg = $maxg")
        if maxg < TOL
            println("Converged in $i iterations.")
            break
        end

        K = end_assemble(_K)

        ΔΔu = cholfact(Symmetric(K[u_free,u_free], :U))\g[u_free]
        Δu -= ΔΔu
        # return Δu

    end


    return Δu
end
