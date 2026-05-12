import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuMP
using Ipopt
using JSON

function solve_rosenbrock()
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x)
    @variable(model, y)
    @NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)
    optimize!(model)
    return Dict(
        "x" => value(x),
        "y" => value(y),
        "objective" => objective_value(model),
        "termination_status" => string(termination_status(model)),
        "primal_status" => string(primal_status(model)),
    )
end

output_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "results", "raw", "jump_rosenbrock.json")
result = solve_rosenbrock()
open(output_path, "w") do io
    JSON.print(io, result, 2)
end
println("JuMP Rosenbrock result written to $(output_path)")
