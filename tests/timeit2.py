import time

def cyc_timeit(funcs, number=100000, display=True):
    timings = [0] * len(funcs)
    for _ in range(number):
        for index, func in enumerate(funcs):
            start = time.perf_counter()
            func()
            duration = time.perf_counter() - start
            timings[index] += duration

    # average = [x/number for x in timings]

    if display:
        print('Timings')
        for func, timing in zip(funcs, timings):
            print(func.__name__, timing)

    # print('Average')
    # for func, x in zip(funcs, average):
    #     print(func.__name__, x)

    return timings

def cyc_timeit_range_plot(funcs, K, start=0, end=100, step=5, number=1000):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    axes = fig.add_subplot(111)

    lines = axes.plot(*([0] * 2 * len(funcs)))
    for line, func in zip(lines, funcs):
        line.set_label(func.__name__)
        line.set_xdata([])
        line.set_ydata([])

    plt.draw()
    plt.legend()
    for k in range(start, end, step):
        K[0] = k
        timings = cyc_timeit(funcs, number=number, display=False)
        for line, timing in zip(lines, timings):
            line.set_ydata(list(line.get_ydata()) + [timing])
            line.set_xdata(list(line.get_xdata()) + [k])
        
    
    axes.relim()
    axes.autoscale_view(True,True,True)
    plt.pause(0.00000000000001)
    
    input()