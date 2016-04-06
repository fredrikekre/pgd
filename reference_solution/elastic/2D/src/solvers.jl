# FE solvers for displacement
function solveDisplacementField(u,u_mesh,u_free,u_fe_values,u_mp,b::Vector=zeros(2))
    n_u_free_dofs = length(u_free)
    Δu = zeros(n_u_free_dofs)
    u_tri = zeros(u)

    i = -1
    while true; i += 1
        copy!(u_tri,u)
        u_tri[u_free] += Δu

        g = u_residual(u_tri,u_mesh,u_free,u_fe_values,u_mp,b)
        tol = 1e-7
        if maximum(abs(g)) < tol
            # println("Displacement field, u, converged in $i iterations, r = $(maximum(abs(g))).")
            break
        end

        K = u_jacobian(u_tri,u_mesh,u_free,u_fe_values,u_mp,b)
        ΔΔu = cholfact(Symmetric(K, :U))\g
        Δu -= ΔΔu

        # return Δu # Since I know its linear
    end

    return Δu
end
