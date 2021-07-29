using ArgParse
ENV["GKSwstype"]="nul"
ENV["GRDIR"]="";
using Plots; gr()

struct Fault
    kind::Char
    addr::Int64
    time::Int64
end


function parse_input(inf::String, t)

    h::Int64 = 0
    m::Int64 = 0
    p::Int64 = 0
    e::Int64 = 0

    local faults::Vector{Fault}
    iranges::Vector{Int64} = [0]
    sizehint!(iranges, 4) # at least 3 i guess, who cares
    r::Int32 = 0
    hits = Set{Int64}()

    GC.enable(false)
    begin
        println(stderr, "read file")
        @time odata = read(inf, String)
        begc::Int64 = 1

        println(stderr, "Find ranges")
        while odata[begc] == '0'
            endc::Int64 = findnext(isequal('\n'), odata, begc+1)
            r += 1
            saddr::SubString = split(SubString(odata, begc, endc-1), ",")[2]
            push!(iranges, parse(Int64, saddr) + last(iranges))
            println("Range:", last(iranges))
            begc = endc+1
        end

        println(stderr, "count lines")
        nlength = countlines(IOBuffer(odata)) - r

        println(stderr, "prealloc")
        bi = 0
        @time faults = Vector{Fault}(undef, nlength)
        @time batchid = Vector{Int64}(undef, nlength)
        sizehint!(hits, nlength)

        odata_len = length(odata)
        println(stderr, "foreach line assignment")
        index_offset = 0
        most_recent_time = 0
        for index::Int64 in 1:nlength
            index -= index_offset
            #endc::Int64 = findnext(isequal(','), odata, begc)
            endc::Int64 = findnext(isequal('\n'), odata, begc)
            @inbounds kind::Char = odata[begc]
            # count each line type; skip B/S markers 
            if kind != 's' && kind != 'b'
                tokens = split(SubString(odata, begc, endc-1), ",")
                #println(tokens)
                #@inbounds
                if (length(tokens) > 2)
                    most_recent_time = parse(Int64, tokens[3])
                end
                @inbounds faults[index] = Fault(odata[begc],  parse(Int64, tokens[2], base=16) >> 12, most_recent_time)
                batchid[index] = bi
                #@inbounds faults[index] = Fault(odata[begc],  parse(Int64, SubString(odata, begc+2, endc-1), base=16) >> 12)
                #@inbounds 
                fault::Fault = faults[index]
                begc = endc + 1
                if fault.kind == 'p'
                    if !(fault.addr in hits)
                        push!(hits, fault.addr)
                        p += 1
                    end
                elseif fault.kind == 'f'
                    if fault.addr in hits
                        h += 1
                    else
                        push!(hits, fault.addr)
                        m += 1
                    end
                elseif fault.kind == 'e'
                    e += 1
                    if fault.addr in hits
                        delete!(hits, fault.addr)
                    end
                end
            else
                begc = endc + 1 
                index_offset += 1
                if kind == 's'
                    bi += 1
                end
            end
        end
        println(stderr, "Resize")
        #shrink to refit for missing  b/s markers
        @time resize!(faults, nlength - index_offset)
    end

    GC.enable(true)
    
    #println(stderr, "totalf: ", f)

    println(stderr, "allocate y arrays")
    #hit miss predict
    #faddrs64::Vector{Int64} = Vector{Int64}(undef, f)
    haddrs64::Vector{Int64} = Vector{Int64}(undef, h)#zeros(Int64, h)
    maddrs64::Vector{Int64} = Vector{Int64}(undef, m)#zeros(Int64, m)
    paddrs64::Vector{Int64} = Vector{Int64}(undef, p)#zeros(Int64, p)
    eaddrs64::Vector{Int64} = Vector{Int64}(undef, e)#zeros(Int64, e)

    println(stderr, "allocate x arrays")

    #hit miss predict
    #fx::Vector{Int64} = zeros(Int64, f)
    hx::Vector{Int64} = zeros(Int64, h)
    mx::Vector{Int64} = zeros(Int64, m)
    px::Vector{Int64} = zeros(Int64, p)
    ex::Vector{Int64} = zeros(Int64, e)
    i::Int64 = 0
    h = 0
    m = 0
    p = 0
    e = 0


    lko=0
    hits = Set{Int64}()
    sizehint!(hits, length(faults))
    GC.enable(false)
    
    println(stderr, "fill arrays")
    if t
        time_min = minimum(f->f.time, faults)
        @time for fault in faults
             if fault.kind == 'p'
                 if !(fault.addr in hits)
                    push!(hits, fault.addr)
                    @inbounds paddrs64[p+1] = fault.addr
                    @inbounds px[p+1] = fault.time - time_min
                    p += 1
                end
             elseif fault.kind == 'f'
                #i += 1
                #@inbounds faddrs64[f+1] = fault.addr
                #@inbounds fx[f+1] = fault.time - time_min
                #f += 1
                i += 1
                if fault.addr in hits
                    @inbounds haddrs64[h+1] = fault.addr
                    @inbounds hx[h+1] = fault.time - time_min
                    h += 1
                else
                   # should this line be here?
                    push!(hits, fault.addr)
                    @inbounds maddrs64[m+1] = fault.addr
                    @inbounds mx[m+1] = fault.time - time_min
                    m += 1
                end
            else
                @inbounds eaddrs64[e+1] = fault.addr
                @inbounds ex[e+1] = fault.time - time_min
                e += 1
                if fault.addr in hits
                    delete!(hits, fault.addr)
                end
            end
        end
    else

        @time for fault in faults
            i += 1
            if fault.kind == 'p'
                # idk if this is appropriate for oversubscribe
                if !(fault.addr in hits)
                    push!(hits, fault.addr)
                    @inbounds paddrs64[p+1] = fault.addr
                    @inbounds px[p+1] = batchid[i]
                    p += 1
                end
            elseif fault.kind == 'f'
                if fault.addr in hits
                    @inbounds haddrs64[h+1] = fault.addr
                    @inbounds hx[h+1] = batchid[i]
                    h += 1
                else
                   # should this line be here?
                    push!(hits, fault.addr)
                    @inbounds maddrs64[m+1] = fault.addr
                    @inbounds mx[m+1] = batchid[i]
                    m += 1
                end
            else
                @inbounds eaddrs64[e+1] = fault.addr
                @inbounds ex[e+1] = batchid[i]
                e += 1
                if fault.addr in hits
                    delete!(hits, fault.addr)
                end
            end
        end
        #=
        @time for fault in faults
            #if fault.kind == 'f'
            #    i += 1
            #    @inbounds faddrs64[f+1] = fault.addr
            #    @inbounds fx[f+1] = i
            #    f += 1
            #end
            lko += 1
            #println("fault: ", lko, " ", length(faults), " ", fault.kind)
            if fault.kind == 'p'
                # idk if this is appropriate for oversubscribe
                if !(fault.addr in hits)
                    push!(hits, fault.addr)
                    @inbounds paddrs64[p+1] = fault.addr
                    @inbounds px[p+1] = i
                    p += 1
                end
            elseif fault.kind == 'f'
                i += 1
                if fault.addr in hits
                    @inbounds haddrs64[h+1] = fault.addr
                    @inbounds hx[h+1] = i
                    h += 1
                else
                   # should this line be here?
                    push!(hits, fault.addr)
                    @inbounds maddrs64[m+1] = fault.addr
                    @inbounds mx[m+1] = i
                    m += 1
                end
            else
                @inbounds eaddrs64[e+1] = fault.addr
                @inbounds ex[e+1] = i
                e += 1
                if fault.addr in hits
                    delete!(hits, fault.addr)
                end
            end
        end
        =#
    end 
    GC.enable(true)

    #return faddrs64, fx, iranges
    return haddrs64, maddrs64, paddrs64, eaddrs64, hx, mx, px, ex, iranges
end

#function adjust_offsets(ranges, fulladdrs, faddrs64)
function adjust_offsets(ranges, fulladdrs, haddrs64, maddrs64, paddrs64, eaddrs64)

    offsets::Vector{Int64} = []
    @inline function scalar_red(val, thresh, off)
        return ifelse(val > thresh, val - off, val)
    end


    GC.enable(false)
    println(stderr, "range adjust")
    @time for r in ranges
        r = Int64(ceil(r))
        redset = filter(x-> x > r-1, fulladdrs)
        if isempty(redset)
            println(stderr, "no distance in range")
            push!(offsets,0)
        else
            push!(offsets, minimum(redset) - r )
            fulladdrs=Set(scalar_red(a, r, last(offsets)) for a in fulladdrs)
        end
    end

    println(stderr, "offsets:", offsets)
    threshs::Vector{Int64} = []
    offs::Vector{Int64} = []
    for (r, o) in zip(ranges, offsets)
        if o != 0
            push!(threshs, r)
            push!(offs, o)
        end
    end
    GC.enable(true)

    @inline function red(val)
        #TODO Thread.threads? How slow is this for "real" problems? Should i use text2norm first anyway?
        for (thresh::Int64, off::Int64) in zip(threshs, offs)
            val -= ifelse(val > thresh, off, 0)
        end
        return val
    end

    @assert(length(threshs) == length(offs))
    println(stderr, "haddrs adjust")
    GC.enable(false)
    #@time faddrs::Vector{Int32} = Vector{Int32}(undef, length(faddrs64)) #map(red, haddrs64)
    #faddrs .= red.(faddrs64)
    @time haddrs::Vector{Int32} = Vector{Int32}(undef, length(haddrs64)) #map(red, haddrs64)
    haddrs .= red.(haddrs64)

    println(stderr, "maddrs adjust")
    @time maddrs::Vector{Int32} = Vector{Int32}(undef, length(maddrs64)) #map(red, maddrs64)
    maddrs .= red.(maddrs64)

    println(stderr, "paddrs adjust")
    @time paddrs::Vector{Int32} = Vector{Int32}(undef, length(paddrs64)) #map(red, paddrs64)
    paddrs .= red.(paddrs64)

    println(stderr, "eaddrs adjust")
    @time eaddrs::Vector{Int32} = Vector{Int32}(undef, length(eaddrs64)) #map(red, eaddrs64)
    eaddrs .= red.(eaddrs64)
    GC.enable(true)

    return haddrs, maddrs, paddrs, eaddrs
    #return faddrs

end

function build_dataset(inf, t)
    #@time faddrs64, fx, iranges = parse_input(inf, t)
    @time haddrs64, maddrs64, paddrs64, eaddrs64, hx, mx, px, ex, iranges = parse_input(inf, t)#, STATS_ONLY)
    println(stderr, "Parse input")
    mint::Int64 = typemax(Int64) #0x7FFFFFFFFFFFFFFF
    #addrmin = min(min(paddrs), min(haddrs), min(maddrs)) 
    #
    #addrmin = minimum(faddrs64)
    addrmin = min(minimum(isempty(paddrs64) ? [mint] : paddrs64), 
                  minimum(isempty(haddrs64) ? [mint] : haddrs64), 
                  minimum(isempty(maddrs64) ? [mint] : maddrs64), 
                  minimum(isempty(eaddrs64) ? [mint] : eaddrs64))
    println(stderr, "faddrs")
    #faddrs64 .-= addrmin
    paddrs64 .-= addrmin
    println(stderr, "haddrs")
    haddrs64 .-= addrmin
    println(stderr, "maddrs")
    maddrs64 .-= addrmin
    eaddrs64 .-= addrmin

    ranges::Vector{Float64} = [i / 4096 for i in iranges]

    #println(stderr, "Num outputs: ", length(faddrs64))

    println(stderr, "fulladdrs")
    fulladdrs = Set{Int64}()
    #sizehint!(fulladdrs, length(faddrs64))
    sizehint!(fulladdrs, length(paddrs64) + length(haddrs64) + length(maddrs64) + length(eaddrs64))
    #union!(fulladdrs, Set{Int64}(faddrs64))
    union!(fulladdrs, Set{Int64}(paddrs64), Set{Int64}(haddrs64), Set{Int64}(maddrs64), Set{Int64}(eaddrs64))

    # try to adjust pages to fill gaps between ranges...
    pop!(ranges)
    popfirst!(ranges)
    
    println(stderr, "ranges: ", ranges)
    println(stderr, "min, max, len: ", minimum(fulladdrs), ", ", maximum(fulladdrs), ", ", length(fulladdrs))
    
    #@time faddrs = adjust_offsets(ranges, fulladdrs, faddrs64)
    @time haddrs, maddrs, paddrs, eaddrs = adjust_offsets(ranges, fulladdrs, haddrs64, maddrs64, paddrs64, eaddrs64)
    println(stderr, "adjust_offsets")
    
    # does this actually need to be reconstructed? probably 
    # also this is rly just debug info
    println(stderr, "Verify input now that it's organized")
    @time empty!(fulladdrs)
    #sizehint!(fulladdrs, length(faddrs))
    sizehint!(fulladdrs, length(paddrs) + length(haddrs) + length(maddrs) + length(eaddrs))
    #@time union!(fulladdrs, Set{Int64}(faddrs))
    @time union!(fulladdrs, Set{Int64}(paddrs), Set{Int64}(haddrs), Set{Int64}(maddrs), Set{Int64}(eaddrs))
    #@time for i in 0:maximum(fulladdrs)-1
    #    if i in fulladdrs
    #    else
    #        println(stderr, "######MISSING:", i)
    #        #println(stderr, "and maybe more but stopping here")
    #        #break
    #    end
    #end
    println(stderr, "min, max, len: ", minimum(fulladdrs), ", ", maximum(fulladdrs), ", ", length(fulladdrs))
    
    #return faddrs, fx, ranges
    return haddrs, maddrs, paddrs, eaddrs, hx, mx, px, ex, ranges

end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "files"
        help = "input file path"
        required = true
        nargs = '+'
        "-s"
        help = "marker size"
        default = 1.
        arg_type = Float64
        "-m"
        help="marker"
        default=","
        "-f"
        action = :store_true
        help="full: display redundant faults"
        "-n"
        help = "out file name"
        default = ""
        "-b"
        help = "VAblock only"
        action = :store_true
        "-t"
        help = "time instead of order"
        action = :store_true
    end
    args = parse_args(ARGS, s)

    markersize = args["s"]
    figname = args["n"]
    time = args["t"]

    ps::Vector{Plots.Plot} = []
    for inf in args["files"]

        fname = splitext(basename(inf))[1]

    #    @time faddrs, fx, ranges = build_dataset(inf, time)
        @time haddrs, maddrs, paddrs, eaddrs, hx, mx, px, ex, ranges = build_dataset(inf, time) #, STATS_ONLY)
        println(stderr, "build_dataset finished")
        if args["b"]
            #faddrs .= faddrs .÷ 512
            #faddrs .= faddrs .* 512
            haddrs .= haddrs .÷ 512
            haddrs .= haddrs .* 512
            maddrs .= maddrs .÷ 512
            maddrs .= maddrs .* 512
            paddrs .= paddrs .÷ 512
            paddrs .= paddrs .* 512
            eaddrs .= eaddrs .÷ 512
            eaddrs .= eaddrs .* 512
        end


        if isempty(figname)
            figname = "pattern.png"
        end

        GC.enable(false)
        @time h = plot(ranges, color=:black, seriestype=:hline, markersize=.1, linewidth=.1, label="")
        GC.enable(true)
        println(stderr, "hlines")
        GC.enable(false)
        title=string(rsplit(inf, '/')[3])
        if occursin("reg-nofetch", inf)
            title="Regular Access"
        elseif occursin("rand-nofetch", inf)
            title="Random Access"
        elseif occursin("cublas", inf)
            title="sgemm"
        end


        @show typeof(title)
        @show string(title)
        tfs=22
        gfs=16
        tickfs=12
        if time
            hx::Vector{Float64} = [Float64(t) for t in hx]
            @time p = scatter!(hx, haddrs,
                    xlabel="Time",
                    ylabel="Page Index of Fault",
                    label="",
                    #title=title,
                    markersize=markersize,
                    titlefontsize=tfs,
                    guidefontsize=gfs,
                    tickfontsize=tickfs,
                    linestyle=:dot,
                    color = :green,
                    yformatter = :plain,
                    #xformatter = :plain,
                    grid = false,
                    #xformatter = x->string(Int(x/1e10),"*pow10")
                    ) 
            #TODO fixme if you ever want actual misses this whole chunk needs tofixing FIXME
            @time p = scatter!(mx, maddrs, color=:green, markersize=markersize, markerstrokewidth=0, label="Faults")
        else
            @time p = scatter!(hx, haddrs,
                    xlabel="Batch ID",
                    ylabel="Page Index of Fault",
                    label="",
                    #title=title,
         #ylabelrotation=90,
         #title="fault something",
         #dpi=800,
         #size=(1920 ÷ 2, 1080 ÷ 2),
                    markersize=markersize,
                    titlefontsize=tfs,
                    guidefontsize=gfs,
                    tickfontsize=tickfs,
                    linestyle=:dot,
         #markerstrokewidth=0,
                    color = :green,
         #label = "Faults",
                    yformatter = :plain,
                    xformatter = :plain,
                    grid = false,
                    ) #layout=(1, 1),legend=false)
            @time p = scatter!(mx, maddrs, color=:green, markersize=markersize, markerstrokewidth=0, label="Faults")
        end
        if !isempty(ex)
            @time p = scatter!(ex, eaddrs, color=:purple, markersize=markersize, markerstrokewidth=0, label="Evictions")
           println(stderr, "evicts")
        end

        push!(ps, p)
        @show inf

        GC.enable(true)
    end

    if length(ps) == 6
        l = @layout [a b; c d; e g]
        @show ps
        p = plot(ps..., layout=l, 
                     dpi=800,
                     size=(1920 ÷ 2, 1080),
                     markersize=markersize,
                     markerstrokewidth=0,)
    elseif length(ps) == 8
        l = @layout [a b; c d; e f; g h]
        @show ps
        p = plot(ps..., layout=l, 
                     dpi=800,
                     size=(1920 ÷ 2, Int(1080 * 1.25)),
                     markersize=markersize,
                     markerstrokewidth=0,
                margin=3Plots.mm)

    else
        legend = :best
        if (occursin("gauss", args["files"][1]))
            legend=:bottomleft
        elseif (occursin("hpgmg", args["files"][1]))
            legend=:right
        end
        @show ps
        p = plot(ps..., #layout=l, 
                     dpi=500,
                     framestyle = :box,
                     legend = legend,
                     legendfontsize=10,
                     background_color_legend = nothing,
                     #size=(1920 ÷ 2, 1080 ÷ 2),
                     markersize=markersize,)
            
                     #markerstrokewidth=0, margin=3Plots.mm)
    end
    #plot(ps..., layout=length(ps))

    rm(figname, force=true)
    println(stderr, "Saving figure...", figname)
    @time savefig(figname)
end

main()
