using ArgParse
ENV["GKSwstype"]="nul"
using Plots; gr()

struct Fault
    kind::Char
    addr::Int64
end


function parse_input(inf::String, FULL, STATS_ONLY)

    h::Int32 = 0
    m::Int32 = 0
    p::Int32 = 0
    e::Int32 = 0

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
            endc::Int64 = findnext(isequal('\n'), odata, begc)
            r += 1
            saddr::SubString = split(SubString(odata, begc, endc-1), ",")[2]
            push!(iranges, parse(Int64, saddr) + last(iranges))
            println("Range:", last(iranges))
            begc = endc+1
        end

        println(stderr, "count lines")
        nlength = countlines(IOBuffer(odata)) - r

        println(stderr, "prealloc")
        sizehint!(hits, nlength)
        hits = Set{Int64}()
        @time faults = Vector{Fault}(undef, nlength)

        odata_len = length(odata)
        println(stderr, "foreach line assignment")
        index_offset = 0
        for index::Int64 in 1:nlength
            index -= index_offset
            endc::Int64 = findnext(isequal('\n'), odata, begc)
            @inbounds kind::Char = odata[begc]
            # count each line type; skip B/S markers 
            if kind != 's' && kind != 'b'

                tokens = split(SubString(odata, begc, endc-1), ",")
                @inbounds faults[index] = Fault(odata[begc],  parse(Int64, tokens[2], base=16) >> 12)
                #@inbounds faults[index] = Fault(odata[begc],  parse(Int64, SubString(odata, begc+2, endc-1), base=16) >> 12)
                @inbounds fault::Fault = faults[index]
                begc = endc + 1
                if fault.kind == 'p'
                    if FULL ||  !(fault.addr in hits)
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
            end
        end
        println(stderr, "Resize")
        #shrink to refit for missing  b/s markers
        @time resize!(faults, nlength - index_offset)
        empty!(hits)
    end

    GC.enable(true)
    
    println("hits, misses, preds, evicts, totalf: ", h, ",", m, ",", p, ",", e, ",", h+m)

    if STATS_ONLY
        exit()
    end

    println(stderr, "allocate y arrays")
    #hit miss predict
    haddrs64::Vector{Int64} = Vector{Int64}(undef, h)#zeros(Int64, h)
    maddrs64::Vector{Int64} = Vector{Int64}(undef, m)#zeros(Int64, m)
    paddrs64::Vector{Int64} = Vector{Int64}(undef, p)#zeros(Int64, p)
    eaddrs64::Vector{Int64} = Vector{Int64}(undef, e)#zeros(Int64, e)

    println(stderr, "allocate x arrays")
    #hit miss predict
    hx::Vector{Int32} = zeros(Int32, h)
    mx::Vector{Int32} = zeros(Int32, m)
    px::Vector{Int32} = zeros(Int32, p)
    ex::Vector{Int32} = zeros(Int32, e)

    i::Int32 = 0
    h = 0
    m = 0
    p = 0
    e = 0

    sizehint!(hits, length(faults))
    
    GC.enable(false)
    
    println(stderr, "fill arrays")
    @time for fault in faults
        if fault.kind == 'p'
            # idk if this is appropriate for oversubscribe
            if FULL || !(fault.addr in hits)
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
    GC.enable(true)

    return haddrs64, maddrs64, paddrs64, eaddrs64, hx, mx, px, ex, iranges
end

function adjust_offsets(ranges, fulladdrs, haddrs64, maddrs64, paddrs64, eaddrs64)

    offsets::Vector{Int64} = []
    @inline function scalar_red(val, thresh, off)
        #return ifelse(val > thresh, val - off, val)
        if val > thresh
            return val - off
        else 
            return val
        end
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

end

function build_dataset(inf, FULL, STATS_ONLY)
    @time haddrs64, maddrs64, paddrs64, eaddrs64, hx, mx, px, ex, iranges = parse_input(inf, FULL, STATS_ONLY)
    println(stderr, "Parse input")
    mint::Int64 = typemax(Int64) #0x7FFFFFFFFFFFFFFF
    #addrmin = min(min(paddrs), min(haddrs), min(maddrs)) 
    #
    addrmin = min(minimum(isempty(paddrs64) ? [mint] : paddrs64), 
                  minimum(isempty(haddrs64) ? [mint] : haddrs64), 
                  minimum(isempty(maddrs64) ? [mint] : maddrs64), 
                  minimum(isempty(eaddrs64) ? [mint] : eaddrs64))
    println(stderr, "paddrs")
    paddrs64 .-= addrmin
    println(stderr, "haddrs")
    haddrs64 .-= addrmin
    println(stderr, "maddrs")
    maddrs64 .-= addrmin
    eaddrs64 .-= addrmin

    ranges::Vector{Float64} = [i / 4096 for i in iranges]

    println(stderr, "Num outputs: ", length(paddrs64) + length(maddrs64) + length(haddrs64))
    println(stderr, "Num faults: ", length(maddrs64) + length(haddrs64))

    println(stderr, "fulladdrs")
    fulladdrs = Set{Int64}()
    sizehint!(fulladdrs, length(paddrs64) + length(haddrs64) + length(maddrs64) + length(eaddrs64))
    union!(fulladdrs, Set{Int64}(paddrs64), Set{Int64}(haddrs64), Set{Int64}(maddrs64), Set{Int64}(eaddrs64))

    # try to adjust pages to fill gaps between ranges...
    pop!(ranges)
    popfirst!(ranges)
    
    println(stderr, "ranges: ", ranges)
    println(stderr, "min, max, len: ", minimum(fulladdrs), ", ", maximum(fulladdrs), ", ", length(fulladdrs))
    
    @time haddrs, maddrs, paddrs, eaddrs = adjust_offsets(ranges, fulladdrs, haddrs64, maddrs64, paddrs64, eaddrs64)
    println(stderr, "adjust_offsets")
    
    # does this actually need to be reconstructed? probably 
    # also this is rly just debug info
    println(stderr, "Verify input now that it's organized")
    @time empty!(fulladdrs)
    sizehint!(fulladdrs, length(paddrs) + length(haddrs) + length(maddrs) + length(eaddrs))
    @time union!(fulladdrs, Set{Int64}(paddrs), Set{Int64}(haddrs), Set{Int64}(maddrs), Set{Int64}(eaddrs))
    @time for i in 0:maximum(fulladdrs)-1
        if i in fulladdrs
        else
            println(stderr, "######MISSING:", i)
            #println(stderr, "and maybe more but stopping here")
            #break
        end
    end
    println(stderr, "min, max, len: ", minimum(fulladdrs), ", ", maximum(fulladdrs), ", ", length(fulladdrs))
    
    return haddrs, maddrs, paddrs, eaddrs, hx, mx, px, ex, ranges

end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "file"
        help = "input file path"
        required = true
        "-s"
        help = "marker size"
        default = 2
        "-m"
        help="marker"
        default=","
        "-f"
        action = :store_true
        help="full: display redundant faults"
        "-n"
        help = "out file name"
        default = ""
        "--so"
        help = "stats only"
        action = :store_true
        "--po"
        help= "no prefetch"
        action = :store_false
    end
    args = parse_args(ARGS, s)

    markersize = args["s"]
    inf = args["file"]
    fname = splitext(basename(inf))[1]
    # i think this just lumps hits and misses together
    FULL = args["f"]
    STATS_ONLY = args["so"]
    figname = args["n"]
    PREFETCH= args["po"]

    @time haddrs, maddrs, paddrs, eaddrs, hx, mx, px, ex, ranges = build_dataset(inf, FULL, STATS_ONLY)
    println(stderr, "build_dataset")


    if isempty(figname)
        if FULL
            figname = "prefetch_full.png"
        else
            figname = "prefetch.png"
        end
        #figname = ifelse(FULL, "prefetch_full.png", figname = "prefetch.png")
    end

    @show typeof(hx)
    @show typeof(haddrs)
    GC.enable(false)
    @time plot!(ranges, color=:black, seriestype=:hline, label="")
    GC.enable(true)
    println(stderr, "hlines")
    GC.enable(false)

    if PREFETCH
    @time p = scatter!(hx, haddrs,
             xlabel="Fault Occurence",
             ylabel="Fault Index",
             #ylabelrotation=90,
             title="fault something",
             dpi=800,
             size=(1920, 1080),
             markersize=markersize,
             markerstrokewidth=0,
            color = :green,
            label = "Hits",
            yformatter = :plain,
            xformatter = :plain,
             ) #layout=(1, 1),legend=false)

        println(stderr, "hits")
        @time p = scatter!(mx, maddrs, color=:red, markersize=markersize, markerstrokewidth=0, label="Misses")
        println(stderr, "misses")
        @time p = scatter!(px, paddrs, color=:blue, markersize=markersize, markerstrokewidth=0, label="Predictions")
        println(stderr, "preds")
    else
    
    @time p = scatter!([hx;mx], [haddrs;maddrs],
             xlabel="Fault Occurence",
             ylabel="Page Index",
             #ylabelrotation=90,
             title="sgemm Evictions",
             legend=:topleft,
             dpi=1000,
             #size=(1920, 1080),
             markersize=markersize,
             markerstrokewidth=0,
            color = :green,
            label = "Faults",
            yformatter = :plain,
            xformatter = :plain,
             ) #layout=(1, 1),legend=false)

        println(stderr, "hits")
        println(stderr, "hits")
#        @time p = scatter!(mx, maddrs, color=:green, markersize=markersize, markerstrokewidth=0)
    end

    if !isempty(ex)
        @time p = scatter!(ex, eaddrs, color=:purple, markersize=markersize, markerstrokewidth=0, label="Evictions")
       println(stderr, "evicts")
    end

    GC.enable(true)
    rm(figname, force=true)
    println(stderr, "Saving figure...", figname)
    @time savefig(figname)
end

main()
