using Plots; gr()
p = plot(rand(Int32, 5), rand(Int32, 5))
p = plot(rand(Float64, 5), rand(Float64, 5))
savefig("precompile.png")
