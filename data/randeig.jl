using StatsBase
function randeig(t, n; xlims=(-2.1*√n, 2.1*√n), bins=100)
    const xgrid = linspace(xlims..., bins)
    counts = @parallel (+) for _ = 1:t
        fit(Histogram, eigvals(Symmetric(randn(n, n))), xgrid).weights
    end
    Histogram(xgrid, counts)
end
