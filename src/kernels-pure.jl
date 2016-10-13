#kernels_pure.jl

function hpl2(A, b)
    F = LinAlg.generic_lufact!(A) #Pure Julia implementation, NO LAPACK OR BLAS calls
    F \ b
end
