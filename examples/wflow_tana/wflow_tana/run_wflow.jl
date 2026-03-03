using Wflow

if length(ARGS) == 0
    println("No config file provided.")
    exit(1)
end

toml_path = ARGS[1]
println("Running Wflow with config: $toml_path")

Wflow.run(toml_path)
