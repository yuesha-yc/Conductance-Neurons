module Analysis

using JSON3

function _json_to_data(x)
    if x isa JSON3.Object
        d = Dict{String,Any}()
        for (k,v) in pairs(x)
            d[String(k)] = _json_to_data(v)
        end
        return d
    elseif x isa JSON3.Array
        return [_json_to_data(v) for v in x]
    else
        return x
    end
end

struct ExperimentHandle
    root::String
    runpath::String  # outputs/<rel>/<run_id>
end


end # module
