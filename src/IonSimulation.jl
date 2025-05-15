module IonSimulation

import Luna: PhysData, Maths, Ionisation, Tools, Logging, Fields
import PyPlot: plt, pygui
import FFTW
using Plots
import Statistics
import LinearAlgebra: mul!, ldiv!, inv
using HDF5
using Dates
import Hankel
import SpecialFunctions: besselj, gamma, erf


mutable struct SpacetimeGrid
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
    ω = FFTW.fftshift(Maths.fftfreq(t))

    kz = zeros((Nt, Nr))
    for (ridx, kri) in enumerate(kr)
        for (ωidx, ωi) in enumerate(ω)
            kzsq = ωi^2 / PhysData.c^2 - kri^2 # calculte kz^2 for each value in k-space
            if kzsq > 0 # No imaginary k vectors 
                kz[ωidx, ridx] = sqrt(kzsq) #calculates kz form kz^2
            end
        end
    end
    SpacetimeGrid(R, r, Nr, δr, kr, t, δt, ω, kz, q)
end
mutable struct Efield{nT, ffT}
    E::Array{nT, 2}
    Emod::Array{ComplexF64, 1}
    FT::ffT
    grid
    λ0::Float64
    linop
    zR::Float64
end

function create_efield(r::Vector{Float64}, grid::SpacetimeGrid, w0, λ0, P, fwhm; flags = FFTW.MEASURE)
    ω0 = PhysData.wlfreq(λ0)
    I = 2 * P / (π * w0^2)
    A0 = Tools.intensity_to_field(I) 
    E = zeros(ComplexF64, (size(grid.t)[1], grid.Nr))
    fwhm_I = 1.665*fwhm/2.355 # From field to power
    for (ridx, ri) in enumerate(r)
        for (tidx, ti) in enumerate(grid.t)
            gausbeam =  exp(-(ri^2) / w0^2) #* exp(1im  * (ω0 / PhysData.c))
            gauspulse =  @. Maths.gauss(ti; fwhm = fwhm_I, x0 = 0) * exp(1im * ω0 .* ti) 
            E[tidx, ridx] = @. A0 * gauspulse * gausbeam 
            
        end
    end
    linop =@. -1im*(grid.kz - grid.ω / PhysData.c)
    #linop = @. -1im*(grid.ω* grid.r^2/2*PhysData.c)
    zR = π*w0^2 / λ0
    fft = FFTW.plan_fft(E, 1; flags)
    Emod = similar(E[1, :])
    Efield(E, Emod, fft, grid, λ0, linop, zR)
end
function create_efield(grid::SpacetimeGrid, w0, λ0, P, fwhm)
    create_efield(grid.r, grid, w0, λ0, P, fwhm)
end
function transform_Efield!(field::Efield; backtransform = false, timetransform = true)
    for i in 1:size(field.E)[1]
        if backtransform
             ldiv!(field.Emod, field.grid.q, field.E[i, :])
             field.E[i, :] = field.Emod
         else
             mul!(field.Emod, field.grid.q, field.E[i, :])
             field.E[i, :] = field.Emod
         end
    end
    if timetransform
        if backtransform
            field.E .= field.FT \ field.E
        else
            field.E .= field.FT * field.E
        end
    end
    
 end


function (field::Efield)(; τ = 0, ϕ = [])
    transform_Efield!(field)
    if τ !== 0
        field.E .*= exp.(1im * FFTW.fftshift(field.grid.ω) .* τ)
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
function propagate(field::Efield, z::Number)
    transform_Efield!(field)
    @. field.E *= exp(field.linop*z)*exp(-1im*atan(z/field.zR))
    transform_Efield!(field, backtransform = true)                                                 #Inverse Fourier-transform
end

function add_lens(field::Efield, f::Number)
    #k = 2π/field.λ0
    #phase_factor = exp.(-1im .* k.*field.grid.r.^2 ./(2*f))
    #for i in 1:length(field.E[:,1])
    #    field.E[i, :] .*= phase_factor
    #end

    FFTW.fft!(field.E, 1)
    for (i, omega) in enumerate(field.grid.ω)
        k = omega/PhysData.c
        #  
        phase_factor = exp.(-1im .* k.*field.grid.r.^2 ./(2*f))
        field.E[i, :] .*= phase_factor
    end
    FFTW.ifft!(field.E, 1)
end
function propagate_coll(field, z::Number)
    transform_Efield!(field)
    field.E .*= exp.(-1im *field.grid.kz .* z)
    transform_Efield!(field, backtransform = true)
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
function MakeMask(r::Array{Float64}, OutR::Float64, InR::Float64; plotim = false, taper_strength = 10)    
    Mask = zeros(ComplexF64, length(r))
    for (ridx, ri) in enumerate(r)
        if ri <= OutR &&
               ri >= InR
                Mask[ridx] = 1.0  
        end
    end
    ## find transition positions
    deltamask = [real(Mask[x+1])-real(Mask[x]) for x in 1:length(Mask)-1] 
    extrema_counter = count(x -> x==1.0 || x==-1.0, deltamask)
    δr = r[2]-r[1]
    k = taper_strength
    if extrema_counter == 1
        if  count(x -> x==1.0, deltamask) == 1 # A Mask that is 1 from the start and falls down to 0 at OutR distance (InR = 0.0)
            transition_point = argmax(deltamask)
            right0, right1 = r[transition_point] - k*δr, r[transition_point] + k*δr
            left0, left1 = 0.0, 0.0
            Mask = Maths.planck_taper(r, left0, left1, right0, right1)
        else # A Mask that is 0 from r = 0 to InR and from their to r[end] is 1 (OutR = Inf)
            transition_point = argmin(deltamask)
            right0, right1 = r[end], r[end]
            left0, left1 = r[transition_point] - k*δr, r[transition_point] + k*δr
            Mask = Maths.planck_taper(r, left0, left1, right0, right1)
        end
    elseif extrema_counter == 2 # A Mask that is 0 from the start till InR, than continues to be 1 till OutR after which it is 0 again till the end of the grid.
        left_transition_point = argmax(deltamask)
        right_transition_point = argmin(deltamask)
        left0, left1 = r[left_transition_point] - k*δr, r[left_transition_point] + k*δr
        right0, right1 = r[right_transition_point] - k*δr, r[right_transition_point] + k*δr
        Mask = Maths.planck_taper(r, left0, left1, right0, right1)
    else # A Mask with "wrong" numbers that are outside of the grid
        error("Given Mask Boundries are outside of the grid range!")
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
function ApplyMask(fund_field::Efield,  b, f; taper_strength = 10, plotim = false)
    delayfield = deepcopy(fund_field)
    field = deepcopy(fund_field)
    if plotim == true
        fig, (ax1, ax12, ax13, ax2, ax22, ax23) = plt.subplots(3,2)
    end

    Er = dropdims(sum(abs.(field.E); dims=1); dims=1)
    Er_sum = similar(Er)
    for i in collect(range(1, length(Er)))
        Er_sum[i] = sum(Er[1:i])
    end
    wzidx = argmin(abs.(Er .- b * maximum(Er)))
    radius = field.grid.r[wzidx]
    InnerMask = MakeMask(delayfield.grid.r, radius, 0.0, taper_strength = taper_strength)
    OuterMask = MakeMask(field.grid.r, field.grid.R, radius, taper_strength = taper_strength)

    if plotim == true
        Ir = dropdims(sum(abs.(field.E); dims=1); dims=1)
        Ir ./= maximum(Ir)

        ax1.plot(field.grid.r.*1e4, Ir, label = "Radial Electric Field")
        ax1.plot(field.grid.r.*1e4, InnerMask, label = "Inner Mask")
        ax1.plot(field.grid.r.*1e4, OuterMask, label = "Outer Mask")
        #ax1.axvline(wz)
        ax1.set_xlabel("r [mm]")
        ax1.set_ylabel("Normalised electric-field strength")
        ax1.set_title("Fundamental-Beam")
        ax1.legend() 

        ax2.plot(field.grid.r.*1e4, Ir, label = "Radial Electric Field")
        ax2.plot(field.grid.r.*1e4, InnerMask, label = "Inner Mask")
        ax2.plot(field.grid.r.*1e4, OuterMask, label = "Outer Mask")
        ax2.set_xlabel("r [mm]")
        ax2.set_ylabel("Normalised electric-field strength")
        ax2.set_title("Delay-Beam")
        ax2.legend()
             
    end
    for i in 1:length(field.grid.t)
        field.E[i, :] .*= OuterMask
        delayfield.E[i, :] .*= InnerMask
    end
    if plotim == true
        Ir = dropdims(sum(abs.(field.E); dims=1); dims=1)
        Irdelay = dropdims(sum(abs.(delayfield.E); dims=1); dims=1)

        Imax = max(maximum(Ir), maximum(Irdelay))
        Irdelay ./= Imax
        Ir ./= Imax

        ax12.plot(field.grid.r.*1e4, Ir, label = "After Mask is applied")
        ax12.set_xlabel("r [mm]")
        ax12.set_ylabel("Normalised electric-field strength")
        #ax12.set_title("After Mask")
        ax12.legend()

        ax22.plot(field.grid.r.*1e4, Irdelay, label = "After Mask is applied")
        ax22.set_xlabel("r [mm]")
        ax22.set_ylabel("Normalised electric-field strength")
        #ax22.set_title("After Mask")
        ax22.legend()
        
    end
    if plotim == true
        Ir = dropdims(sum(abs.(field.E); dims=1); dims=1)
        Irdelay = dropdims(sum(abs.(delayfield.E); dims=1); dims=1)

        Imax = max(maximum(Ir), maximum(Irdelay))
        Irdelay ./= Imax
        Ir ./= Imax

        ax13.plot(field.grid.r.*1e4, Ir, label = "In focus")
        ax13.set_xlabel("r [mm]")
        ax13.set_ylabel("Normalised electric-field strength")
        #ax13.set_title("In focus")
        ax13.legend()

        ax23.plot(field.grid.r.*1e4, Irdelay, label = "In focus")
        ax23.set_xlabel("r [mm]")
        ax23.set_ylabel("Normalised electric-field strength")
        #ax23.set_title("In focus")
        ax23.legend()
        plt.tight_layout()
               
    end
    #Ir = dropdims(sum(abs2.(field.E); dims=1); dims=1)
    #Irdelay = dropdims(sum(abs2.(delayfield.E); dims=1); dims=1)
    #fraction = abs(sum(Irdelay)/sum(Ir))
    return delayfield, field
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
    ratio::Number
end
"""
    CreateScan(λ0, PeakP,  fwhm, w0, f, θ, Grid, Mask, p::Propagator, rate)

Constructs a `Scan` for running autocorrelation delay scans.

# Arguments
- `λ0::Number` : Central wavelength
- `PeaP::Number` : Peak power of the pulse.
- `fwhm::Number` : Full-width-half-maximum of the pulse.
- `w0::Number` : Beam waist.
- `f::Number` : f-number of the focusing optics.
- `Θ::Number` : Angle under which the beam propagates.
- `Grid::SpacetimeGrid` : Grid that defines the time and space of interest. Use Grid() to create a grid.
- `Mask::Array{size Grid[x,y]}` : Mask that will be applied to the beam at position -f.
                                  The mask should have the same size has the x,y space in the Grid. The mask will simply be
                                  multiplied to the signal.
- `rate::Ionrate` : Ionisation rate calulated with the LUNA/Ionisation ionrate_fun function. 
                    (e.g ppt = Ionisation.ionrate_fun!_PPTcached(:He, λ0))

"""
 function create_scan(λ0::Number, PeakP:: Number, fwhm::Number, w0::Number, f::Number, r::Number,
    ϕ::Array{Float64}, Grid, rate; coll_mirr_dist = 1,  plotim = false, taper_strength = 10)
    fundamental_field = create_efield(Grid.r, Grid, w0, λ0, PeakP, fwhm)
    fundamental_field(ϕ = ϕ)
    if plotim
        plot_pulse(fundamental_field, title = "Original Pulse")
    end
    propagate(fundamental_field, coll_mirr_dist)
    if plotim
        plot_pulse(fundamental_field, title = "At Lens position")
    end
    add_lens(fundamental_field, -coll_mirr_dist)
    propagate(fundamental_field, 0.1)
    delayfield, field = ApplyMask(fundamental_field, r, f, plotim = plotim, taper_strength = taper_strength)
    propagate(field, 0.1)
    propagate(delayfield, 0.1)
    if plotim
        plot_pulse(field, title = "Field at focusing lens position")
        plot_pulse(delayfield, title = "Delayfield at focusing lens position")
    end
    
    add_lens(field, -f)
    add_lens(delayfield, -f)
    propagate(field, f)
    propagate(delayfield, f)
    if plotim
        plot_pulse(field, title = "Field in Focus")
        plot_pulse(delayfield, title = "Delayfield in Focus")
    end
    Scan(PeakP, fwhm, f, ϕ, w0, rate, field, delayfield, r)
end


function create_delayset(τrange, δτ)
    τsteps = round(Int, 2τrange/δτ)
    if  iseven(τsteps)
        delay = collect(range(start = -τrange, stop = τrange, length = τsteps + 1))
    else
        delay = collect(range(start = -τrange, stop = τrange, length = τsteps))
    end
end

function plot_pulse(field::Efield; r_xlim = (0, field.grid.r[end].*1e3), t_xlim = (-field.grid.t[1].*1e15, field.grid.t[1].*1e15), title = "Title")
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(field.grid.r.*1e3, dropdims(sum(abs2.(field.E[:, :]), dims =1), dims =1))
    plt.xlim(r_xlim)
    plt.xlabel("Radial Distance r [mm]")
    plt.ylabel("Intensity [a.u]")
    plt.title(title)
    plt.subplot(1,2,2)
    plt.plot(field.grid.t.*1e15, sum(field.E[:, :], dims =2))
    plt.xlim(t_xlim)
    plt.xlabel("Time t [fs]")
    plt.ylabel("Intensity [a.u]")
    
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
function (dscan::Scan)(New_R::Float64, new_m::Int64, delayset::Array{Float64}, zset::Array{Float64}, fpath::String, fname::String)
    z = zset
    IonMap = zeros((length(delayset)))
    k = 1
    τsteps = length(delayset)
    zsteps = length(zset)
    create_new_grid(dscan.field, R = New_R, m = new_m)
    create_new_grid(dscan.delayfield, R = New_R, m = new_m)
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
            dscan.delayfield.E .= deepcopy(Edel_fund)
            dscan.field.E .= deepcopy(Efield_foc)
        end
        IonMap[k] = sum(abs, IonMapdummy .* (stop-start)/zsteps)
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
            parnames = ["Efield", "λ", "PeakP", "fwhm", "w0", "f", "ϕ", "τ", "zrange", "Ratio"]
            parvalues = [dscan.field.E, dscan.field.λ0, dscan.PeakP, dscan.fwhm, dscan.w0, dscan.f, dscan.ϕ, delayset, z, dscan.ratio]
            for (key, values) in zip(parnames, parvalues)
                g[key] = values 
            end
            
        end
        
    end
    delayset, IonMap
end

function create_new_grid(field::Efield; R = nothing, m= nothing, δt= nothing, Nt= nothing)
    if m !== nothing
        Nr = 2^m
    else
        Nr = field.grid.Nr
    end
    if isnothing(δt)
        δt = field.grid.δt
        Nt = length(field.grid.t)
    end
    new_grid = FreePolarGrid(R, Nr, δt, Nt)
    E_save = []
    for i in 1:Nt
        cs= Maths.CSpline(field.grid.r, field.E[i, :])
        push!(E_save, cs.(new_grid.r))
    end
    field.grid = new_grid
    field.E = field.E[:, 1:Nr]
    for i in 1:Nt
        field.E[i, :] = E_save[i]
    end
    field.linop =@. -1im*(field.grid.kz - field.grid.ω / PhysData.c)
    field.FT = FFTW.plan_fft(field.E, 1; flags = FFTW.MEASURE)
    field.Emod = similar(field.E[1, :])
end
end


