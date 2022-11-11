using IonSimulation
using Documenter

DocMeta.setdocmeta!(IonSimulation, :DocTestSetup, :(using IonSimulation); recursive=true)

makedocs(;
    modules=[IonSimulation],
    authors="Joleik Nordmann <jmn2000@hw.ac.uk> and contributors",
    repo="https://github.com/JoleikNord/IonSimulation.jl/blob/{commit}{path}#{line}",
    sitename="IonSimulation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JoleikNord.github.io/IonSimulation.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoleikNord/IonSimulation.jl",
    devbranch="master",
)
