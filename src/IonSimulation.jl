module IonSimulation
#__precompile__()
import Luna: PhysData, Maths, Ionisation, Tools
import PyPlot: plt, pygui
import FFTW
using Plots
import Statistics
import LinearAlgebra: mul!, ldiv!, inv
################# Creating a grid modeled after the Luna free grid module ######################
abstract type SpacetimeGrid end
struct Grid <: SpacetimeGrid
    x::Vector{Float64}
    Nx::Int64
    δx::Float64
    y::Vector{Float64}
    Ny::Int64
    δy::Float64
    kx::Vector{Float64}
    ky::Vector{Float64}
    t::Vector{Float64}
    δt::Float64
    ω::Vector{Float64}
    kz::Array{Float64, 3}
    r::Array{Float64, 3}                          # Not used for now
    xywin::Array{Float64, 3}                      # Not used for now
end

function FreeGrid(Rx, Nx, Ry, Ny, δt, Nt; window_factor=0.1)
    Rxw = Rx * (1 + window_factor)
    Ryw = Ry * (1 + window_factor)

    δx = 2Rxw/Nx
    nx = collect(range(0, length=Nx))
    x = @. (nx-Nx/2) * δx
    kx = 2π*FFTW.fftfreq(Nx, 1/δx)

    δy = 2Ryw/Ny
    ny = collect(range(0, length=Ny))
    y = @. (ny-Ny/2) * δy
    ky = 2π*FFTW.fftfreq(Ny, 1/δy)

    nt = collect(range(0, length = Nt))
    t = @. (nt-Nt/2) * δt
    ω = Maths.fftfreq(t)

    kz = zeros((Nt, Nx, Nx))
    for (xidx, kxi) in enumerate(kx)
        for (yidx, kyi) in enumerate(ky)
            for (ωidx, ωi) in enumerate(ω)
                kzsq = ωi^2 / PhysData.c^2 - kxi^2 - kyi^2 # calculte kz^2 for each value in k-space
                if kzsq > 0 # No imaginary k vectors 
                    kz[ωidx, xidx, yidx] = sqrt(kzsq) #calculates kz form kz^2
                end
            end
        end
    end

    r = sqrt.(reshape(y, (1, Ny)).^2 .+ reshape(x, (1, 1, Nx)).^2)

    xwin = Maths.planck_taper(x, -Rxw, -Rx, Rx, Rxw)
    ywin = Maths.planck_taper(y, -Ryw, -Ry, Ry, Ryw)
    xywin = reshape(xwin, (1, length(xwin))) .* reshape(xwin, (1, 1, length(xwin)))


    Grid(x, Nx, δx, y, Ny, δy, kx, ky, t, δt, ω, kz, r, xywin);
end

Grid(R, N, δT, NT) = FreeGrid(R, N, R, N, δT, NT)
grid = Grid(320e-6, 64, 50e-18, 1024)

######################## Creating gausbeam #################################

"""
    Efoc(Grid, w0, λ0, P, fwhm; τ=0, θ=0)

X,Y-electric field in focus for time frame given by the grid size. 

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest
- `w0::Real` : Beam waste in focus
- `λ0::Real` : Central wavelength
- `Power::Real` : Total power transmitted by the beam
- `Full-width-half-maximum::Real` : Full-width-half-maximum of the beam.
- `τ::Real` : Delay of the beam to time 0. The center of the gaussian pulse is moved by -τ
- `θ::Real` : Angle of propagation of the beam. θ defines the wavefront tilt.
"""
function Efoc(Grid::SpacetimeGrid, w0, λ0, P, fwhm, τ, θ)
    ω0 = PhysData.wlfreq(λ0)
    I = 2 * P / (π * w0^2)
    A0 = Tools.intensity_to_field(I) 
    degtorad = θ * pi/180
    Efield = zeros(ComplexF64, (size(Grid.t)[1], size(Grid.x)[1], size(Grid.y)[1]))
    for (xidx, xi) in enumerate(Grid.x)
        for (yidx, yi) in enumerate(Grid.y)
            for (tidx, ti) in enumerate(Grid.t)
                gausbeam =  exp(-(xi^2 + yi^2) / w0^2) * exp(1im * sin(degtorad) * xi * (ω0 / PhysData.c))
                gauspulse =  Maths.gauss(ti.-τ; fwhm = fwhm) * exp(1im * ω0 * (ti.-τ))
                Efield[tidx, xidx, yidx] = @. A0 * gauspulse * gausbeam 
            end
        end
    end
    Efield;
end

function Efoc(Grid::SpacetimeGrid, w0, λ0, P, fwhm, θ)
    Efoc(Grid::SpacetimeGrid, w0, λ0, P, fwhm, 0, θ)
end
function Efoc(Grid::SpacetimeGrid, w0, λ0, P, fwhm)
    Efoc(Grid::SpacetimeGrid, w0, λ0, P, fwhm, 0,0)    
end

##################### Propagating the pulse ###########################
struct Propagator{nT, ftT}
    Ek::Array{nT, 3}
    FT::ftT
    linop::Array{nT, 3}
    zR::Float64
end
"""
    Propagator(Grid, E, w0, λ0; flags=FFTW.MEASURE)

Propagator for propagation of an electric field in space. To propagate the field
first create a propogator with this function and then use it to propagate a field e
by a distance z. (e.g If the propagator object is called p, p(E, z)) 

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest
- `Electric field::Array{Complex64, 3}` : Electric field in space and time created by the "Efoc" function
- `w0::Real` : Beam waste in focus
- `λ0::Real` : Central wavelength
- `flags::???` : FFTW plan for the electric field array.
"""
function Propagator(Grid::SpacetimeGrid, E, w0, λ0; flags=FFTW.MEASURE)
    FT = FFTW.plan_fft(E; flags)
    Ek = similar(E)
    linop = @. 1im*(Grid.kz - Grid.ω / PhysData.c)
    zR = π*w0^2 / λ0
    Propagator(Ek, FT, FFTW.fftshift(linop, 1), zR)
end 

function (p::Propagator)(E, z)
    mul!(p.Ek, p.FT, E)
    @. p.Ek *= exp(p.linop*z)*exp(-1im*atan(z/p.zR)) 
    p.FT \ p.Ek                                                  #Inverse Fourier-transform
end

############### Ionisation calculation #######################
"""
    Ionfrac(Grid, rate, E)

Calculates the fraction of ionisation induced by an electric field. 

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest
- `rate::???` : Ionisation rate calulated with the LUNA/Ionisation ionrate_fun function. 
                (e.g ppt = Ionisation.ionrate_fun!_PPTcached(:He, λ0))
- `Electric field::Array{Complex64, 3 or 4}` : Electric field in space and time created by the "Efoc" function.
                                               also works for propagation cubes.
"""
function Ionfrac(Grid::SpacetimeGrid, rate, E::Array{ComplexF64,3})
    ppt_ionfrac = similar(E[1, :, :])
    for i in eachindex(ppt_ionfrac[1, :])
        for j in eachindex(ppt_ionfrac[:, 1])
            ppt_ionfrac[i, j] = Ionisation.ionfrac(rate, real(E[:, i, j]), Grid.δt)[end]
        end
    end
    ppt_ionfrac
end
function Ionfrac(Grid::SpacetimeGrid, rate, E::Array{ComplexF64,4})
    ppt_ionfrac = similar(E[:, 1, :, :])
    for i in eachindex(ppt_ionfrac[:, 1, 1])
        for j in eachindex(ppt_ionfrac[1, :, 1])
            for k in eachindex(ppt_ionfrac[1, 1, :])
                ppt_ionfrac[i, j, k] = Ionisation.ionfrac(rate, real(E[i, :, j, k]), Grid.δt)[end]
            end
        end
    end
    ppt_ionfrac;

end
#################### Creating & applying beam masks ########################
"""
    MakeMask(Grid, OutR, InR)

Creates a mask that can be applyed to block part of a beam.

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest
- `OuterRadius::Float64` : Outer radius of the mask in m. 
- `InnerRadius::Float64` : Inner radius of the mask in m.
"""
function MakeMask(Grid::SpacetimeGrid, OutR::Float64, InR::Float64)
    Mask = ones(ComplexF64, (Grid.Nx, Grid.Ny))
    for xi ∈ 1:Grid.Nx
        for yi ∈ 1:Grid.Ny
            if (float(xi) * Grid.δx - (Grid.Nx * Grid.δx / 2) - Grid.δx)^2 + (float(yi) * Grid.δy - (Grid.Ny * Grid.δy / 2) - Grid.δy)^2 <= OutR^2 &&
               (float(xi) * Grid.δx - (Grid.Nx * Grid.δx / 2) - Grid.δx)^2 + (float(yi) * Grid.δy - (Grid.Ny * Grid.δy / 2) - Grid.δy)^2 >= InR^2
                Mask[xi, yi] = 0              # Checks for each element in the x-y array of the grid if the distance of the element
                                              # to the center of the array is larger than the outer and smaller than the inner radius
                                              # and if not sets the mask value to 0. Otherwise the value becomes 1.
            end
        end
    end
    Mask;
end
"""
    ApplyMask!(E, Mask, f, p::Propagator)

Applys the given mask to an electric field.

# Arguments
- `Electric field::Array{Complex64, 3}` : Electric field in space and time created by the "Efoc" function.
- `Mask::Array{Int64, 3}` : The mask that will be applyed to the field. It can be created using the "MakeMask" function. 
- `focal length f::Float64` : The focal length of the system. The electric field will be propagated backwards at a distance -f
                              then the mask will be applied and the masked beam will be propagated forwards back to the focus.
- `Propagator p::Propagator` : The Propagator that will be used to propagate the electric field E. To create a Propagator use the "Propagator" function.     
- `Plot Imagae::Boolean` : If true will plot an image of the beam right after the masked has been applyed. Usefull to check the mask size.                
"""
function ApplyMask(Grid::SpacetimeGrid, E, Mask, f, p::Propagator, plotim = false)
    Em = p(E, -f)
    for i ∈ 1:size(E, 1)
        Em[i, :, :] .*= Mask
    end
    if plotim == true
        PlotPulse(Grid, Em)
    end
    E = p(Em, f);
end

################### Creating a 4-dim propagation cube #############################
"""
    PropCube(p::Propagator, E, zrange, Nz)

Creates a 4-D propagation cube with Nz electric field slices in zrange.

# Arguments
- `Propagator p::Propagator` : The Propagator that will be used to propagate the electric field E. 
                               To create a Propagator use the "Propagator" function. 
- `Electric field::Array{Complex64, 3}` : Electric field in space and time created by the "Efoc" function.
- `zrange::Tuple{Float64, Float64}` : The range in z direction in which the propagation cube will span. 
- `Nz::Int64` : Number of electric field slices that will be simulated in the given zrange.
"""
function PropCube(p::Propagator, E, zrange::Tuple{Float64, Float64}, Nz::Int64)
    start, stop = zrange
    PropCube =zeros(ComplexF64, (Nz, size(E)[1], size(E)[2], size(E)[3]))
    for i in collect(range(0; length = Nz - 1))
        PropCube[i+1, :, :, :] = p(E, start + ((stop - start) / Nz) * i)
        print("Step $i of $Nz \n")
    end
    PropCube;
end
## Plotting and animation functions --------------------------------------------
function PlotPulse(Grid::SpacetimeGrid, E)
    plt.figure()
    plt.pcolormesh(Grid.x, Grid.y, abs2.(E[1, :, :]))
    plt.colorbar()
    plt.gcf()
end

function Animate(Grid::SpacetimeGrid, Ecube, fps, fname::String)
    frames = size(Ecube, 1)
    anim_dummy = @animate for i ∈ 1:frames
       #fig = plot(y * 1e6, abs2.(PulseCube[i, Nt÷2, Nx÷2+1, :]),c = :viridis, ylims = (0, 1e20))
       fluence = dropdims(sum(abs2.(Ecube[i, :, :, :]); dims=1); dims=1)
       fig = heatmap(Grid.x * 1e6, Grid.y * 1e6, fluence,c = :viridis)
   end
   gif(anim_dummy, fname * ".gif", fps = fps)
end
#Currently under construction
#=
mutable struct Scan{nT}
    λ::Float64
    PeakP:: Float64
    fwhm::Float64
    w0::Float64
    f::Float64
    θ::Float64
    τ::Float64
    Grid::SpacetimeGrid
    Mask::Array{ComplexF64, 2}
    E::Array{nT, 3}
    p::Propagator
    rate::Any
    folname::String
    fname::String
end


function CreateScan(λ0::Float64, PeakP:: Float64,  fwhm::Float64, w0::Float64, f::Float64, θ::Float64, τ::Float64, Grid::SpacetimeGrid, Mask::Array{ComplexF64, 2},p::Propagator,rate,  folname::String,fname::String)
    E = Efoc(Grid, w0, λ0, PeakP, fwhm, τ, θ)
    Emask = ApplyMask(Grid, E, Mask, f, p)
    Scan(λ0, PeakP,  fwhm, w0, f, θ, τ, Grid, Mask, Emask, p, rate, folname,fname)
end
function (dscan::Scan)(DRange::Tuple{Float64, Float64}, dsteps::Int64, dmask::Array{ComplexF64, 2})
    IonMap = []
    E = copy(dscan.E)
    start, stop = DRange
    delay = collect(range(start, stop, dsteps))
    for i ∈ delay
        dscan.E = Efoc(dscan.Grid, dscan.w0, dscan.λ, dscan.PeakP, dscan.fwhm, i, dscan.θ)
        dscan.E = ApplyMask(dscan.Grid, dscan.E, dmask, dscan.f, dscan.p)
        dscan.E += E
        push!(IonMap, mean(Ionfrac(Gird, dscan.rate, dscan.E)))
        printnl("Step $i from $delay")
    end
    IonMap
end
##

function DelayScan(Grid::SpacetimeGrid, w0, λ0, P, fwhm, Ionrate, delrange::Tuple, delsteps::Int64, Mask, Maskdel, f, Prop, θ, θsteps = 0, sfol = " ")
    delay = collect(range(delrange[1], delrange[2], delsteps))
    if θ isa(Tuple)
        angles = collect(range(θ[1], θ[2], θsteps))
        for j in angles
            k = 0
            Delayscanner.E = Efoc(Grid, w0, λ0, P, fwhm, j)
            Delayscanner.E = ApplyMask(Grid, Delayscanner.Ek, Mask, f, Prop)
            for i ∈ delay
                Delayscanner.Ed = Efoc(Grid, w0, λ0, P, fwhm, i, -j)
                Delayscanner.Ed = ApplyMask(Grid, Delayscanner.Ed, Maskl, f, p)
                Delayscanner.Ed = Delayscanner.E + Delayscanner.Ed
                Iondelay = Ionfrac(Grid, Ionrate, Delayscanner.Ed)
                push!(Delayscanner.IonMap, mean(Delayscanner.Iondelay))
                k += 1
                println("Step $k of $delaysteps")
            end
            @save "IonMap_dshape_angle=$j.jld2" delay IonMap

            plt.plot(delay, Delayscanner.IonMap, label = "Crossing angle = ${round(j, digits = 2)}")
            plt.legend()
            plt.gcf()
end
=#

