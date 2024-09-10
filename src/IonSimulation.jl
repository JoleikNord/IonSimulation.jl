import Luna: PhysData, Maths, Ionisation, Tools, Logging, Fields
import PyPlot: plt, pygui
import FFTW
using Plots
import Statistics
import LinearAlgebra: mul!, ldiv!, inv
using HDF5
using Dates
import Hankel
import SpecialFunctions: besselj, gamma

pygui(true)


abstract type SpacetimeGrid end
struct RealGrid <: SpacetimeGrid
    R::Float64
    r::Vector{Float64}
    Nr::Int64
    δr::Float64
    kr::Vector{Float64}
    t::Vector{Float64}
    δt::Float64
    ω::Vector{Float64}
    kz::Array{Float64, 2}
    q::Any        
end
function FreePolarGrid(R, Nr, δt, Nt; window_factor=0.1)
    #Rxw = Rx * (1 + window_factor)
    #Ryw = Ry * (1 + window_factor)
    q = Hankel.QDHT(0, R, Nr) # dim=2
    δr = R/Nr
    nR = collect(range(0, length=Nr))
    r = q.r
    #δkr = pi/R
    #kr = δkr .* nR
    kr = q.k
    nt = collect(range(0, length = Nt))
    t = @. (nt-Nt/2) * δt
    ω = Maths.fftfreq(t)

    kz = zeros((Nt, Nr))
    for (ridx, kri) in enumerate(kr)
        for (ωidx, ωi) in enumerate(ω)
            kzsq = ωi^2 / PhysData.c^2 - kri^2 # calculte kz^2 for each value in k-space
            if kzsq > 0 # No imaginary k vectors 
                kz[ωidx, ridx] = sqrt(kzsq) #calculates kz form kz^2
            end
        end
    end
    RealGrid(R, r, Nr, δr, kr, t, δt, ω, kz, q)
end
mutable struct Efield{nT, ffT}
    E::Array{nT, 2}
    Emod::Array{ComplexF64, 1}
    FT::ffT
    grid::SpacetimeGrid
    λ0::Float64
    linop::Array{nT, 2}
    zR::Float64
end

function create_efield(r::Vector{Float64}, grid::SpacetimeGrid, w0, λ0, P, fwhm; flags = FFTW.MEASURE)
    ω0 = PhysData.wlfreq(λ0)
    I = 2 * P / (π * w0^2)
    A0 = Tools.intensity_to_field(I) 
    E = zeros(ComplexF64, (size(grid.t)[1], grid.Nr))
    for (ridx, ri) in enumerate(r)
        for (tidx, ti) in enumerate(grid.t)
            gausbeam =  exp(-(ri^2) / w0^2) * exp(1im  * (ω0 / PhysData.c))
            gauspulse =  @. Maths.gauss(ti; fwhm = fwhm) * exp(1im * ω0 .* ti)
            E[tidx, ridx] = @. A0 * gauspulse * gausbeam 
            
        end
    end
    linop = @. -1im*(grid.kz - grid.ω / PhysData.c)
    zR = π*w0^2 / λ0
    fft = FFTW.plan_fft(E, 1; flags)
    Emod = similar(E[1, :])
    Efield(E, Emod, fft, grid, λ0, linop, zR)
end

function create_efield(grid::SpacetimeGrid, w0, λ0, P, fwhm)
    create_efield(grid.r, grid, w0, λ0, P, fwhm)
end

function transform_Efield!(field::Efield; backtransform = false)
    for i in 1:size(field.E)[1]
       if backtransform
            ldiv!(field.Emod, field.grid.q, field.E[i, :])
            field.E[i, :] = field.Emod
        else
            mul!(field.Emod, field.grid.q, field.E[i, :])
            field.E[i, :] = field.Emod
        end
    end
    if backtransform
        field.E .= field.FT \ field.E
    else
        field.E .= field.FT * field.E
    end
end



function (field::Efield)(; τ = 0, ϕ = [])
    transform_Efield!(field)
    if τ !== 0
        field.E .*= exp.(1im * FFTW.fftshift(field.grid.ω) * τ)
    end
    if !isnothing(ϕ)
        field.E .= FFTW.fftshift(field.E)
        Fields.prop_taylor!(field.E, field.grid.ω, ϕ, field.λ0)
        field.E .= FFTW.ifftshift(field.E)
    end 
    transform_Efield!(field; backtransform = true)
end

"""
    (p::Propagator)(E, z)

Propagets the field `E` by the distance `z`.
"""
function propagate(field::Efield, z::Float64)
    transform_Efield!(field)
    @. field.E *= exp(field.linop*z)*exp(-1im*atan(z/field.zR)) 
    transform_Efield!(field, backtransform = true)                                                 #Inverse Fourier-transform
end


##

############### Ionisation calculation #######################
"""
    Ionfrac(Grid, rate, E)

Calculates the fraction of ionisation induced by an electric field. 

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest
- `rate::ionrate` : Ionisation rate calulated with the LUNA/Ionisation ionrate_fun function. 
                (e.g ppt = Ionisation.ionrate_fun!_PPTcached(:He, λ0))
- `Electric field::Array{Complex64, 3 or 4}` : Electric field in space and time created by the "Efoc" function.
                                               also works for propagation cubes.
"""
function Ionfrac(field::Efield, rate; NumberDensity = 2.5e25, electron_charge = -1.6e-19)
    rate_ionfrac = similar(field.E[1, :])
    for i in eachindex(rate_ionfrac)
            rate_ionfrac[i] = Ionisation.ionfrac(rate, real(field.E[:, i]), field.grid.δt)[end]
    end
    Ne_dz = 2*pi*abs(sum(rate_ionfrac .* field.grid.δr .* field.grid.r))*NumberDensity*electron_charge
    rate_ionfrac, Ne_dz
end


#################### Creating & applying beam masks ########################
"""
    MakeMask(Grid, OutR, InR)

Creates a mask that can be applyed to block part of a beam.

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest.
- `OuterRadius::Float64` : Outer radius of the mask in m. 
- `InnerRadius::Float64` : Inner radius of the mask in m.
"""
function MakeMask(r::Array{Float64}, OutR::Float64, InR::Float64; plotim = false)
    Mask = zeros(ComplexF64, length(r))
    for (ridx, ri) in enumerate(r)
        if ri <= OutR &&
               ri >= InR
                Mask[ridx] = 1  
        end
    end
    if plotim
        plt.figure()
        plt.plot(r*1e3, abs2.(Mask))
        plt.xlabel("r [mm]")
        plt.ylabel("Reflectivity [%]")
        plt.ylim(-0.1, 1.1)
    end
    Mask;
end


"""
    ApplyMask(E, Mask, f, p::Propagator)

Applys the given mask to an electric field.

# Arguments
- `Electric field::Array{Complex64, 3}` : Electric field in space and time created by the "Efoc" function.
- `Mask::Array{Int64, 3}` : The mask that will be applyed to the field. It can be created using the "MakeMask" function. 
- `focal length f::Float64` : The focal length of the system. The electric field will be propagated backwards at a distance -f
                              then the mask will be applied and the masked beam will be propagated forwards back to the focus.
- `Propagator p::Propagator` : The Propagator that will be used to propagate the electric field E. To create a Propagator use the "Propagator" function.     
- `plotim::Boolean` : If true will plot an image of the beam right after the masked has been applyed. Usefull to check the mask size.                
"""
function ApplyMask(field::Efield, Mask, f; plotim = false)
    #θ = M2 * λ0 / (π * w0)
    #beamdia = 2*f *tan(θ + w0)
    #beamratio = beamdia / w0
    #farfield_grid = FreePolarGrid(field.grid.R * beamratio, length(field.grid.r) * ceil(beamratio), 1e-15, 512)
    field.E = propagate(field, -f)
    for i ∈ 1:size(field.E, 1)
        field.E[i, :] .*= Mask
    end
    if plotim == true
        PlotPulse(field)
    end
    field.E = propagate(field, f);
end

function PlotPulse(field::Efield; t = 0) 
    if t !== 0
        ti = t
    else
        ti = length(field.grid.t)÷2
    end

    plt.figure()
    plt.plot(field.grid.r*1e3, abs2.(field.E[ti, :])*1e-4)
    plt.xlabel("r [mm]")
    plt.ylabel("Intensity [W/cm^2]")
    plt.figure()
    plt.pcolormesh(field.grid.r*1e3, field.grid.t*1e15, abs2.(field.E)*1e-4)
    plt.xlabel("r [mm]")
    plt.ylabel("t [fs]")
end

function PropCube(field::Efield, zrange::Tuple{Float64, Float64}, Nz::Int64)
    (start, stop) = zrange
    PropCube = zeros(ComplexF64, (Nz, size(field.E)[1], size(field.E)[2]))
    for i in collect(range(0; length = Nz - 1))
        PropCube[i+1, :, :] = propagate(field, ((stop - start) / Nz))
        print("Step $i of $Nz \n")
    end
    PropCube;
end

function Animate(field::Efield, Ecube, fps, fname::String)
    frames = size(Ecube, 1)
    anim_dummy = @animate for i ∈ 1:frames
       fluence = abs2.(Ecube[i, :, :])
       fig = heatmap(field.grid.r * 1e6, field.grid.t*1e15, fluence, c = :viridis)
   end
   gif(anim_dummy, fname * ".gif", fps = fps)
end



##
struct Scan{}
    PeakP::Float64
    fwhm::Float64
    f::Float64
    ϕ::Array{Float64}
    w0::Float64
    rate::Any
    field::Efield
    delayfield::Efield
end
"""
    CreateScan(λ0, PeakP,  fwhm, w0, f, θ, Grid, Mask, p::Propagator, rate)

Constructs a `Scan` for running autocorrelation delay scans.

# Arguments
- `λ0::Float64` : Central wavelength
- `PeaP::Float64` : Peak power of the pulse.
- `fwhm::Float64` : Full-width-half-maximum of the pulse.
- `w0::Float64` : Beam waist.
- `f::Float64` : f-number of the focusing optics.
- `Θ::Float64` : Angle under which the beam propagates.
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest. Use Grid() to create a grid.
- `Mask::Array{size Grid[x,y]}` : Mask that will be applied to the beam at position -f.
                                  The mask should have the same size has the x,y space in the Grid. The mask will simply be
                                  multiplied to the signal.
- `rate::Ionrate` : Ionisation rate calulated with the LUNA/Ionisation ionrate_fun function. 
                    (e.g ppt = Ionisation.ionrate_fun!_PPTcached(:He, λ0))

"""
 function create_scan(λ0::Float64, PeakP:: Float64, fwhm::Float64, w0::Float64, f::Float64, 
    ϕ::Array{Float64}, Grid, rate)
    field = create_efield(Grid.r, Grid, w0, λ0, PeakP, fwhm)
    field(ϕ = ϕ)
    delayfield = create_efield(Grid.r, Grid, w0, λ0, PeakP, fwhm)
    delayfield(ϕ = ϕ)
    Scan(PeakP, fwhm, f, ϕ, w0, rate, field, delayfield,)
end


function create_delayset(τrange, δτ)
    τsteps = round(Int, 2τrange/δτ)
    if  iseven(τsteps)
        delay = collect(range(start = -τrange, stop = τrange, length = τsteps + 1))
    else
        delay = collect(range(start = -τrange, stop = τrange, length = τsteps))
    end
end
"""
    (dscan::Scan)(drange, dstepts, zrange, zsteps, mask_in, mask_out, fpath, fname)

Runs an autocorrelation ionisation delay scan. 
Arguments
`drange::Tuple{Float64, Float64}` : Delay range
`dsteps::Int64` : Number of delay steps in the given `drange`
`zrange::Tuple{Float64, Float64}` : Dinstance around the focus in which the ionisation signal will be measured
`zsteps::Int64` : Number of steps in the given `zrange` in which the ionisation will be calculated 
`mask_in::Array{size Grid[x,y]}` : Inner mask that will be applied to the beam. For mor info see the functions `MakeMask`.
`mask_out::Array{size Grid[x,y]}` : Outer mask that will be applied to the beam. 
`fpath::String` : Path under which the scan will be saved. In this path a subfolder will be created with the current date as a title.
`fname::String` : Filename of the scan.

If the arguments `zrange` and `zsteps` arent given the scan will be run only in the focus. 
Likewise if the arguments `mask_in` and `mask_out` aren't given the scan will be run with the full beam as signal and delay.
"""
function (dscan::Scan)(delayset::Array{Float64}, zrange::Tuple{Float64, Float64}, zsteps::Int64, InnerMask::Array{ComplexF64, 1}, OuterMask::Array{ComplexF64, 1}, fpath::String, fname::String)
    start, stop = zrange
    z = collect(range(start, stop, zsteps))
    IonMap = zeros((length(delayset)))
    k = 1
    τsteps = length(delayset)
    if !isempty(InnerMask)
        ApplyMask(dscan.delayfield, InnerMask, dscan.f)
    end
    if !isempty(OuterMask)
        ApplyMask(dscan.field, OuterMask, dscan.f)
    end
    Eori_fund = deepcopy(dscan.field.E)
    Edel_fund = deepcopy(dscan.delayfield.E)
    for i ∈ delayset
        dscan.delayfield.E .= Edel_fund
        dscan.field.E .= Eori_fund
        IonMapdummy = zeros(length(z))
        dscan.delayfield(τ = i)
        Edel_foc = deepcopy(dscan.delayfield.E)
        l = 1
        for j ∈ z   
            propagate(dscan.delayfield, j)
            propagate(dscan.field, j)
            dscan.delayfield.E .+= dscan.field.E
            IonMapdummy[l] = Ionfrac(dscan.delayfield, dscan.rate)[2]
            println("Step $l| $k from $zsteps | $τsteps")
            l += 1
            dscan.delayfield.E .= deepcopy(Edel_foc)
            dscan.field.E .= deepcopy(Eori_fund)
        end
        IonMap[k] = sum(abs,IonMapdummy .* (stop-start)/zsteps)
        k += 1
    end
    date = Dates.format(now(), "yyyy-mm-dd")
    folpath = mkpath(joinpath(fpath, date))
    filename = fname*"_"* Dates.format(now(), "HH-MM-SS") * ".h5"
    filepath = joinpath(folpath, filename) 
    if !isfile(filepath)
        HDF5.h5open(filepath, "w") do file
            create_group(file, "data")
            create_group(file, "params")
            f = file["data"]
            g = file["params"]
            #HDF5.create_dataset(f, "Ionisation_Fraction", Float64, 2)
            f["Ionisation_Fraction"] = IonMap
            f["delay"] = delayset 
            parnames = ["Efield", "λ", "PeakP", "fwhm", "w0", "f", "ϕ", "τ", "zrange", "InMask", "OutMask"]
            parvalues = [dscan.field.E, dscan.field.λ0, dscan.PeakP, dscan.fwhm, dscan.w0, dscan.f, dscan.ϕ, delayset, z, InnerMask, OuterMask]
            for (key, values) in zip(parnames, parvalues)
                g[key] = values 
            end
            
        end
    end
    delayset, IonMap
end


