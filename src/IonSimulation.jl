module IonSimulation
#__precompile__()
import Luna: PhysData, Maths, Ionisation, Tools, Logging, Fields
import PyPlot: plt, pygui
import FFTW
using Plots
import Statistics
import LinearAlgebra: mul!, ldiv!, inv
using HDF5
using Dates
################# Creating a grid modeled after the Luna free grid module ######################
abstract type SpacetimeGrid end
struct RealGrid <: SpacetimeGrid
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

    #=f_lims = PhysData.c./λ_lims
    Logging.@info ("Freq limits %.2f - %.2f PHz", f_lims[2]*1e-15, f_lims[1]*1e-15)
    δto = min(1/(6*maximum(f_lims)), δt) # 6x maximum freq, or user-defined if finer
    samples = 2^(ceil(Int, log2(trange/δto))) # samples for fine grid (power of 2)
    trange_even = δto*samples # keep frequency window fixed, expand time window as necessary
    Logging.@info ("Samples needed: %.2f, samples: %d, δt = %.2f as",
                            trange/δto, samples, δto*1e18)
    Logging.@info ("Requested time window: %.1f fs, actual time window: %.1f fs", trange*1e15, trange_even*1e15)
    δωo = 2π/trange_even
    Nto = collect(range(0, length=samples))
    to = @. (Nto-samples/2)*δto # centre on 0
    Nωo = collect(range(0, length=Int(samples/2 +1)))
    ωo = Nωo*δωo

    ωmin = 2π*minimum(f_lims)
    ωmax = 2π*maximum(f_lims)
    ωmax_win = 1.1*ωmax
    cropidx = findfirst(x -> x>ωmax_win, ωo)
    cropidx = 2^(ceil(Int, log2(cropidx))) + 1 # make coarse grid power of 2 as well
    ω = ωo[1:cropidx]
    δt = π/maximum(ω)
    tsamples = (cropidx-1)*2
    Nt = collect(range(0, length=tsamples))
    t = @. (Nt-tsamples/2)*δt
    
    # Make apodisation windows
    ωwindow = Maths.planck_taper(ω, ωmin/2, ωmin, ωmax, ωmax_win) 

    twindow = Maths.planck_taper(t, minimum(t), -trange/2, trange/2, maximum(t))
    towindow = Maths.planck_taper(to, minimum(to), -trange/2, trange/2, maximum(to))

    # Indices to select real frequencies (for dispersion relation)
    sidx = (ω .> ωmin/2) .& (ω .< ωmax_win)

    @assert δt/δto ≈ length(to)/length(t)
    @assert δt/δto ≈ maximum(ωo)/maximum(ω)

    #Logging.@info @sprintf("Grid: samples %d / %d, ωmax %.2e / %.2e",
    #                       length(t), length(to), maximum(ω), maximum(ωo))
    =#
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


    RealGrid(x, Nx, δx, y, Ny, δy, kx, ky, t, δt, ω, kz, r, xywin);
end

FreeGrid(R, N, δt, Nt) = FreeGrid(R, N, R, N, δt, Nt)

######################## Creating gausbeam #################################

"""
    Efoc(Grid, w0, λ0, P, fwhm; τ=0, θ=0)

x,y-electric field in focus for time frame given by the grid size. 

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest
- `w0::Real` : Beam waste in focus
- `λ0::Real` : Central wavelength
- `Power::Real` : Total power transmitted by the beam
- `Full-width-half-maximum::Real` : Full-width-half-maximum of the beam.
- `τ::Real` : Delay of the beam to time 0. The center of the gaussian pulse is moved by -τ
- `θ::Real` : Angle of propagation of the beam. θ defines the wavefront tilt.
- `ϕ::Vector{Real}` : Angle of propagation of the beam. θ defines the wavefront tilt.

"""
function Efoc(Grid::SpacetimeGrid, w0, λ0, P, fwhm; τ = 0, θ = 0, ϕ = [])
    ω0 = PhysData.wlfreq(λ0)
    I = 2 * P / (π * w0^2)
    A0 = Tools.intensity_to_field(I) 
    Efield = zeros(ComplexF64, (size(Grid.t)[1], size(Grid.x)[1], size(Grid.y)[1]))
    for (xidx, xi) in enumerate(Grid.x)
        for (yidx, yi) in enumerate(Grid.y)
            for (tidx, ti) in enumerate(Grid.t)
                gausbeam =  exp(-(xi^2 + yi^2) / w0^2) * exp(1im * sin(θ * pi/180) * xi * (ω0 / PhysData.c))
                gauspulse =  @. Maths.gauss(ti; x0 = -τ, fwhm = fwhm) * exp(1im * ω0 .* (ti-τ))
                Efield[tidx, xidx, yidx] = @. A0 * gauspulse * gausbeam 
                
            end
        end
    end
    if !isempty(ϕ)
        Eω = FFTW.fftshift(FFTW.fft(Efield, 1), 1)
        Eωshift = Fields.prop_taylor!(Eω, Grid.ω, ϕ, λ0)
        Eshift = FFTW.ifft(FFTW.ifftshift(Eωshift, 1), 1)
        Eshift
    else
        Efield
    end
end
##
"""
    Efield
    
Electic field of a gaussian beam in focus.
"""
struct Efield{nT, ffT}
    E::Array{nT, 3}
    Emod::Array{nT, 3}
    FT::ffT
    grid::SpacetimeGrid
    λ0::Float64
end

"""
    Efield(Grid, w0, λ0, P, fwhm; θ=0, flags = FFTW.MEASURE)

Construct a `Efield`. Which can be further modified e.g adding phases or delays. 

# Arguments
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest
- `w0::Real` : Beam waste in focus
- `λ0::Real` : Central wavelength
- `Power::Real` : Total power transmitted by the beam
- `Full-width-half-maximum::Real` : Full-width-half-maximum of the beam.
- `θ::Real` : Angle of propagation of the beam. θ defines the wavefront tilt.
- `flags::fft_plan_method`: args from FFTW.plan_fft  

"""

function Efield(grid::SpacetimeGrid, w0, λ0, P, fwhm; θ = 0, flags = FFTW.MEASURE)
    ω0 = PhysData.wlfreq(λ0)
    I = 2 * P / (π * w0^2)
    A0 = Tools.intensity_to_field(I) 
    E = zeros(ComplexF64, (size(grid.t)[1], size(grid.x)[1], size(grid.y)[1]))
    for (xidx, xi) in enumerate(grid.x)
        for (yidx, yi) in enumerate(grid.y)
            for (tidx, ti) in enumerate(grid.t)
                gausbeam =  exp(-(xi^2 + yi^2) / w0^2) * exp(1im * sin(θ * pi/180) * xi * (ω0 / PhysData.c))
                gauspulse =  @. Maths.gauss(ti; fwhm = fwhm) * exp(1im * ω0 .* ti)
                E[tidx, xidx, yidx] = @. A0 * gauspulse * gausbeam 
                
            end
        end
    end
    fft = FFTW.plan_fft(E, 1; flags)
    Emod = similar(E)
    Efield(E, Emod, fft, grid, λ0)
end
"""
    (field::Efield)(; τ = 0, ϕ = [])

Adds a delay `τ` and/or a phase `ϕ` to the field.
`ϕ` can be a vector for higher order phases to be added.
"""

function (field::Efield)(; τ = 0, ϕ = [])
    mul!(field.Emod , field.FT, field.E)
    if τ !== 0
        field.Emod .*= exp.(1im * field.grid.ω * τ)
    end
    if !isnothing(ϕ)
        field.Emod .= FFTW.fftshift(field.Emod)
        Fields.prop_taylor!(field.Emod, field.grid.ω, ϕ, field.λ0)
        field.Emod .= FFTW.ifftshift(field.Emod)
    end 
    field.E .= field.FT \ field.Emod
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
- `flags::FFTW_plan` : FFTW plan for the electric field array.
"""
function Propagator(Grid::SpacetimeGrid, E, w0, λ0; flags=FFTW.MEASURE)
    FT = FFTW.plan_fft(E; flags)
    Ek = similar(E)
    linop = @. 1im*(Grid.kz - Grid.ω / PhysData.c)
    zR = π*w0^2 / λ0
    Propagator(Ek, FT, FFTW.fftshift(linop, 1), zR)
end 
"""
    (p::Propagator)(E, z)

Propagets the field `E` by the distance `z`.
"""
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
- `rate::ionrate` : Ionisation rate calulated with the LUNA/Ionisation ionrate_fun function. 
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
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest.
- `OuterRadius::Float64` : Outer radius of the mask in m. 
- `InnerRadius::Float64` : Inner radius of the mask in m.
"""
function MakeMask(grid::SpacetimeGrid, OutR::Float64, InR::Float64; plotim = false)
    Mask = ones(ComplexF64, (grid.Nx, grid.Ny))
    for xi ∈ 1:grid.Nx
        for yi ∈ 1:grid.Ny
            rsquare = ((xi - grid.Nx÷2 - 1) * grid.δx)^2 + ((yi - grid.Ny÷2 - 1) * grid.δy)^2
            if rsquare <= OutR^2 &&
               rsquare >= InR^2
                Mask[xi, yi] = 0              # Checks for each element in the x-y array of the grid if the distance of the element
                                              # to the center of the array is larger than the outer and smaller than the inner radius
                                              # and if not sets the mask value to 0. Otherwise the value becomes 1.
            end
        end
    end
    if plotim
        plt.figure()
        plt.pcolormesh(grid.x, grid.y, abs2.(Mask))
        plt.colorbar()
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
- `Plot Imagae::Boolean` : If true will plot an image of the beam right after the masked has been applyed. Usefull to check the mask size.                
"""
function ApplyMask(grid::SpacetimeGrid, E, Mask, f, p::Propagator; plotim = false)
    Em = p(E, -f)
    for i ∈ 1:size(E, 1)
        Em[i, :, :] .*= Mask
    end
    if plotim == true
        PlotPulse(grid, Em)
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
    PropCube = zeros(ComplexF64, (Nz, size(E)[1], size(E)[2], size(E)[3]))
    for i in collect(range(0; length = Nz - 1))
        PropCube[i+1, :, :, :] = p(E, start + ((stop - start) / Nz) * i)
        print("Step $i of $Nz \n")
    end
    PropCube;
end
## Plotting and animation functions --------------------------------------------
function PlotPulse(grid::SpacetimeGrid, E; t = 0) 
    if t !== 0
        ti = t
    else
        ti = length(grid.t)÷2
    end

    plt.figure()
    plt.pcolormesh(grid.x, grid.y, abs2.(E[ti, :, :]))
    plt.colorbar()
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
##
struct Scan{}
    λ::Float64
    PeakP::Float64
    fwhm::Float64
    w0::Float64
    f::Float64
    θ::Float64
    ϕ::Array{Float64}
    Grid::Any
    Eorigin::Efield
    Edelay::Efield
    rate::Any
    p::Propagator
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
 function CreateScan(λ0::Float64, PeakP:: Float64, fwhm::Float64, w0::Float64, f::Float64, 
    θ::Float64, ϕ::Array{Float64}, Grid::SpacetimeGrid, rate)
    Eorigin = Efield(Grid, w0, λ0, PeakP, fwhm, θ = θ)
    Eorigin(ϕ = ϕ)
    Edelay = deepcopy(Eorigin)
    p = Propagator(Grid, Eorigin.E, w0, λ0)
    Scan(λ0, PeakP, fwhm, w0, f, θ, ϕ, Grid, Eorigin, Edelay, rate, p)
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
function (dscan::Scan)(δτ::Float64, τrange::Float64, zrange::Tuple{Float64, Float64}, zsteps::Int64, InnerMask::Array{ComplexF64, 2}, OuterMask::Array{ComplexF64, 2}, fpath::String, fname::String)
    τsteps = round(Int, 2τrange/δτ)
    if  iseven(τsteps)
        delay = collect(range(start = -τrange, stop = τrange, length = τsteps + 1))
    else
        delay = collect(range(start = -τrange, stop = τrange, length = τsteps))
    end

    start, stop = zrange
    z = collect(range(start, stop, zsteps))
    IonMap = zeros(length(delay))
    k = 1
    if !isempty(InnerMask)
        dscan.Edelay.E .= ApplyMask(dscan.Grid, dscan.Edelay.E, InnerMask, dscan.f, dscan.p)
    end
    if !isempty(OuterMask)
        dscan.Eorigin.E .= ApplyMask(dscan.Grid, dscan.Eorigin.E, OuterMask, dscan.f, dscan.p)
    end
    Eori_fund = deepcopy(dscan.Eorigin.E)
    Edel_fund = deepcopy(dscan.Edelay.E)
    for i ∈ delay
        dscan.Edelay.E .= Edel_fund
        dscan.Eorigin.E .= Eori_fund
        IonMapdummy = zeros(length(z))
        dscan.Edelay(τ = i)
        Edel_foc = deepcopy(dscan.Edelay.E)
        l = 1
        for j ∈ z   
            dscan.Edelay.E .= dscan.p(dscan.Edelay.E, j)
            dscan.Eorigin.E .= dscan.p(dscan.Eorigin.E, j)
            dscan.Edelay.E .+= dscan.Eorigin.E
            IonMapdummy[l] = Statistics.mean(Ionfrac(dscan.Grid, dscan.rate, dscan.Edelay.E))
            println("Step $l| $k from $zsteps | $τsteps")
            l += 1
            dscan.Edelay.E .= deepcopy(Edel_foc)
            dscan.Eorigin.E .= deepcopy(Eori_fund)
        end
        IonMap[k] = Statistics.mean(IonMapdummy)
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
            f["delay"] = delay 
            parnames = ["Efield", "λ", "PeakP", "fwhm", "w0", "f", "θ", "ϕ", "τ", "zrange", "InMask", "OutMask"]
            parvalues = [dscan.Eorigin.E, dscan.λ, dscan.PeakP, dscan.fwhm, dscan.w0, dscan.f, dscan.θ, dscan.ϕ, delay, z, InnerMask, OuterMask]
            for (key, values) in zip(parnames, parvalues)
                g[key] = values 
            end
            
        end
    end
    
end
function (dscan::Scan)(drange::Tuple{Float64, Float64}, dsteps::Int64, zrange::Tuple{Float64, Float64}, zsteps::Int64, fpath::String, fname::String,)
    start, stop = drange
    delay = collect(range(start, stop, dsteps))
    start, stop = zrange
    z = collect(range(start, stop, zsteps))
    IonMap = zeros(length(delay))
    k = 1
    for i ∈ delay
        IonMapdummy = zeros(length(z))
        dscan.Edelay = Efoc(dscan.Grid, dscan.w0, dscan.λ, dscan.PeakP, dscan.fwhm, τ = i, θ = dscan.θ, ϕ = dscan.ϕ)
        l = 1
        for j ∈ z    
            dscan.Edelay = dscan.p(dscan.Edelay, j)
            dscan.Eorigin = dscan.p(dscan.Eorigin, j)
            dscan.Edelay += dscan.Eorigin
            IonMapdummy[l] = Statistics.mean(Ionfrac(dscan.Grid, dscan.rate, dscan.Edelay))
            println("Step $l| $k from $zsteps | $dsteps")
            l += 1
        end
        IonMap[k] = Statistics.mean(IonMapdummy)
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
            f["delay"] = delay 
            parnames = ["λ", "PeakP", "fwhm", "w0", "f", "θ", "ϕ", "τ","zrange"]
            parvalues = [dscan.λ, dscan.PeakP, dscan.fwhm, dscan.w0, dscan.f, dscan.θ, dscan.ϕ, delay, z]
            for (key, values) in zip(parnames, parvalues)
                g[key] = values 
            end
            
        end
    end
    
end
function (dscan::Scan)(drange::Tuple{Float64, Float64}, dsteps::Int64, InnerMask::Array{ComplexF64, 2}, OuterMask::Array{ComplexF64, 2}, fpath::String, fname::String)
    start, stop = drange
    delay = collect(range(start, stop, dsteps))
    IonMap = zeros(length(delay))   
    k = 1
    for i ∈ delay
        dscan.Edelay = Efoc(dscan.Grid, dscan.w0, dscan.λ, dscan.PeakP, dscan.fwhm, τ = i, θ = dscan.θ, ϕ = dscan.ϕ)
        if !isempty(InnerMask)
            dscan.Edelay = ApplyMask(dscan.Grid, dscan.Edelay, InnerMask, dscan.f, dscan.p)
        end
        if !isempty(OuterMask)
            dscan.Eorigin = ApplyMask(dscan.Grid, dscan.Eorigin, OuterMask, dscan.f, dscan.p)
        end
        dscan.Edelay += dscan.Eorigin
        IonMap[k] = Statistics.mean(Ionfrac(dscan.Grid, dscan.rate, dscan.Edelay))
        println("Step $k from $dsteps")
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
            f["delay"] = delay 
            parnames = ["λ", "PeakP", "fwhm", "w0", "f", "θ", "ϕ", "τ", "InMask","OutMask"]
            parvalues = [dscan.λ, dscan.PeakP, dscan.fwhm, dscan.w0, dscan.f, dscan.θ, dscan.ϕ, delay, InnerMask, OuterMask]
            for (key, values) in zip(parnames, parvalues)
                g[key] = values 
            end
            
        end
    end
end
function (dscan::Scan)(drange::Tuple{Float64, Float64}, dsteps::Int64, fpath::String, fname::String)
    start, stop = drange
    delay = collect(range(start, stop, dsteps))
    IonMap = zeros(length(delay))   
    k = 1
    for i ∈ delay
        dscan.Edelay = Efoc(dscan.Grid, dscan.w0, dscan.λ, dscan.PeakP, dscan.fwhm, τ = i, θ = dscan.θ, ϕ = dscan.ϕ)
        dscan.Edelay += dscan.Eorigin
        IonMap[k] = Statistics.mean(Ionfrac(dscan.Grid, dscan.rate, dscan.Edelay))
        println("Step $k from $dsteps")
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
            f["delay"] = delay 
            parnames = ["λ", "PeakP", "fwhm", "w0", "f", "θ", "ϕ", "τ"]
            parvalues = [dscan.λ, dscan.PeakP, dscan.fwhm, dscan.w0, dscan.f, dscan.θ, dscan.ϕ, delay]
            for (key, values) in zip(parnames, parvalues)
                g[key] = values 
            end
            
        end
    end
end
end

