# same order * same order
@inline hadamard{order,dim}(S1::Tensor{order,dim}, S2::Tensor{order,dim}) = S1.*S2

# 1st order * 2nd order

@generated function hadamard{dim,T1,T2}(S1::Tensor{1,dim,T1}, S2::Tensor{2,dim,T2})
    N = ContMechTensors.n_components(Tensor{2,dim})
    rows = Int(N^(1/2))
    exps = Expr[]

    for j in 1:rows, i in 1:rows
        I = i
        J = ContMechTensors.compute_index(Tensor{2,dim}, i, j)
        push!(exps,:(S1.data[$I]*S2.data[$J]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta,:inline))
        Tensor{2,dim,typeof(zero(T1)*zero(T2))}($exp)
    end
end

@inline function hadamard{dim}(S1::Tensor{1,dim}, S2::SymmetricTensor{2,dim})
    hadamard(S1,convert(Tensor{2,dim},S2))
end

@generated function hadamard{dim,T1,T2}(S1::Tensor{2,dim,T1}, S2::Tensor{1,dim,T2})
    N = ContMechTensors.n_components(Tensor{2,dim})
    rows = Int(N^(1/2))
    exps = Expr[]

    for j in 1:rows, i in 1:rows
        I = ContMechTensors.compute_index(Tensor{2,dim}, i, j)
        J = j
        push!(exps,:(S1.data[$I]*S2.data[$J]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta,:inline))
        Tensor{2,dim,typeof(zero(T1)*zero(T2))}($exp)
    end
end

@inline function hadamard{dim}(S1::SymmetricTensor{2,dim}, S2::Tensor{1,dim})
    hadamard(convert(Tensor{2,dim},S1), S2)
end

# 2nd order * 4th order

@generated function hadamard{dim,T1,T2}(S1::Tensor{2,dim,T1}, S2::Tensor{4,dim,T2})
    N = ContMechTensors.n_components(Tensor{4,dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        I = ContMechTensors.compute_index(Tensor{2,dim}, i, j)
        J = ContMechTensors.compute_index(Tensor{4,dim}, i, j, k, l)
        push!(exps,:(S1.data[$I]*S2.data[$J]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta,:inline))
        Tensor{4,dim,typeof(zero(T1)*zero(T2))}($exp)
    end
end

@inline function hadamard{dim}(S1::SecondOrderTensor{dim}, S2::FourthOrderTensor{dim})
    hadamard(convert(Tensor{2,dim},S1),convert(Tensor{4,dim},S2))
end

@generated function hadamard{dim,T1,T2}(S1::Tensor{4,dim,T1}, S2::Tensor{2,dim,T2})
    N = ContMechTensors.n_components(Tensor{4,dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        I = ContMechTensors.compute_index(Tensor{4,dim}, i, j, k, l)
        J = ContMechTensors.compute_index(Tensor{2,dim}, k, l)
        push!(exps,:(S1.data[$I]*S2.data[$J]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta,:inline))
        Tensor{4,dim,typeof(zero(T1)*zero(T2))}($exp)
    end
end

@inline function hadamard{dim}(S1::FourthOrderTensor{dim}, S2::SecondOrderTensor{dim})
    hadamard(convert(Tensor{4,dim},S1),convert(Tensor{2,dim},S2))
end


# 2nd * 4th * 2nd

@inline function hadamard{dim}(S1::SecondOrderTensor{dim},S2::FourthOrderTensor{dim}, S3::SecondOrderTensor{dim})
    hadamard(convert(Tensor{2,dim},S1),convert(Tensor{4,dim},S2),convert(Tensor{2,dim},S3))
end

@generated function hadamard{dim,T1,T2,T3}(S1::Tensor{2,dim,T1},S2::Tensor{4,dim,T2}, S3::Tensor{2,dim,T3})
    N = ContMechTensors.n_components(Tensor{4,dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        I = ContMechTensors.compute_index(Tensor{2,dim}, i, j)
        J = ContMechTensors.compute_index(Tensor{4,dim}, i, j, k, l)
        K = ContMechTensors.compute_index(Tensor{2,dim}, k, l)
        push!(exps,:(S1.data[$I]*S2.data[$J]*S3.data[$K]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta,:inline))
        Tensor{4,dim,typeof(zero(T1)*zero(T2)*zero(T3))}($exp)
    end
end


const ∘ = hadamard

########################
# Handle special cases #
########################

# @inline function hadamard{dim1, dim2}(S1::SecondOrderTensor{dim1},S2::FourthOrderTensor{dim2}, S3::SecondOrderTensor{dim1})
#     S1a = convert(ContMechTensors.get_main_type(typeof(S1)){2,dim2}, S1)
#     S3a = convert(ContMechTensors.get_main_type(typeof(S3)){2,dim2}, S3)
#     return hadamard(S1a,S2,S3a)
# end
