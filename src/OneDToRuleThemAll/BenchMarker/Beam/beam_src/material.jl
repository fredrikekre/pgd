function get_E_tensor(E, ν, dim)

    λ = E*ν/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))

    δ = (i,j) -> i == j ? 1.0 : 0.0

    data = typeof(λ+μ)[]

    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        Eijkl = λ * δ(i,j) * δ(k,l) + 2.0 * μ * 0.5 * (δ(i,k) * δ(j,l) + δ(i,l) * δ(j,k))
        push!(data,Eijkl)
    end

    return Tensor{4,dim}((data...))

end
