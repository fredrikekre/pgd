hadamard{order,dim}(S1::Tensor{order,dim}, S2::Tensor{order,dim}) = S1.*S2

function hadamard{dim,T1,T2}(S1::Tensor{1,dim,T1}, S2::Tensor{2,dim,T2})
    T =typeof(zero(T1)*zero(T2))
    data = T[]

    for j in 1:dim, i in 1:dim
        push!(data,S1[i]*S2[i,j])
    end
    return Tensor{2,dim,T}(data)
end

function hadamard{dim,T1,T2}(S1::Tensor{2,dim,T1}, S2::Tensor{1,dim,T2})
    T =typeof(zero(T1)*zero(T2))
    data = T[]

    for j in 1:dim, i in 1:dim
        push!(data,S1[i,j]*S2[j])
    end
    return Tensor{2,dim,T}(data)
end

function hadamard{dim,T1,T2}(S1::Tensor{4,dim,T1}, S2::Tensor{2,dim,T2})
    T =typeof(zero(T1)*zero(T2))
    data = T[]

    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        push!(data,S1[i,j,k,l]*S2[k,l])
    end
    return Tensor{4,dim,T}(data)
end

function hadamard{dim,T1,T2}(S1::Tensor{2,dim,T1}, S2::Tensor{4,dim,T2})
    T =typeof(zero(T1)*zero(T2))
    data = T[]

    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        push!(data,S1[i,j]*S2[i,j,k,l])
    end
    return Tensor{4,dim,T}(data)
end

const ∘ = hadamard
