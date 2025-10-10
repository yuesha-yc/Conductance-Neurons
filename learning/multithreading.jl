Threads.nthreads(:default)

Threads.nthreads(:interactive)

a = zeros(10)

Threads.@threads for i = 1:10
    a[i] = Threads.threadid()
end

a

function sum_single(a)
    s = 0
    for i in a
        s += i
    end
    s
end