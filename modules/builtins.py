#from os import plock
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from scipy import fftpack
import math
from PySpice.Unit import *
from matplotlib.pylab import *

def dumping_oscillation():
        def data_gen(t=0):
            cnt = 0
            while cnt < 1000:
                cnt += 1
                t += 0.1
                yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)

        def init():
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlim(0, 10)
            del xdata[:]
            del ydata[:]
            line.set_data(xdata, ydata)
            return line,

        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.grid()
        xdata, ydata = [], []


        def run(data):
            # update the data
            t, y = data
            xdata.append(t)
            ydata.append(y)
            xmin, xmax = ax.get_xlim()

            if t >= xmax:
                ax.set_xlim(xmin, 2*xmax)
                ax.figure.canvas.draw()
            line.set_data(xdata, ydata)

            return line,
        plt.title('dumping oscillation')

        ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
                                      repeat=False, init_func=init)
        plt.show()

def pulse_generation():
    class Scope(object):
        def __init__(self, ax, maxt=2, dt=0.02):
            self.ax = ax
            self.dt = dt
            self.maxt = maxt
            self.tdata = [0]
            self.ydata = [0]
            self.line = Line2D(self.tdata, self.ydata)
            self.ax.add_line(self.line)
            self.ax.set_ylim(-.1, 1.1)
            self.ax.set_xlim(0, self.maxt)

        def update(self, y):
            lastt = self.tdata[-1]
            if lastt > self.tdata[0] + self.maxt:  # reset the arrays
                self.tdata = [self.tdata[-1]]
                self.ydata = [self.ydata[-1]]
                self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
                self.ax.figure.canvas.draw()

            t = self.tdata[-1] + self.dt
            self.tdata.append(t)
            self.ydata.append(y)
            self.line.set_data(self.tdata, self.ydata)
            return self.line,

    def emitter(p=0.03):
        'return a random value with probability p, else 0'
        while True:
            v = np.random.rand(1)
            if v > p:
                yield 0.
            else:
                yield np.random.rand(1)
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    fig, ax = plt.subplots()
    scope = Scope(ax)
    # pass a generator in "emitter" to produce data for the update func
    ani = animation.FuncAnimation(fig, scope.update, emitter, interval=10,
                                  blit=True)

    plt.show()

def sine_cosine():
    x = np.arange(0,4*np.pi-1,0.1)   # start,stop,step
    y = np.sin(x)
    z = np.cos(x)
    plt.plot(x,y,x,z)
    plt.axhline(y=0, color='black')
    plt.xlabel('x values from 0 to 4pi')  # string must be enclosed with quotes '  '
    plt.ylabel('sin(x) and cos(x)')
    plt.title('Plot of sin and cos from 0 to 4pi')
    plt.legend(['sin(x)', 'cos(x)'])      # legend entries as seperate strings in a list
    plt.show()

def cosine_wave():
    time = np.arange(0, 20, 0.2);
    amplitude   = np.cos(time)
    plt.plot(time, amplitude)
    plt.title('Cosine wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude = cosine(time)')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='b')
    plt.show()

def diode_eq():
    # http://helloworld922.blogspot.com/2014/11/experimenting-with-diodes-and-non.html
    pass

def sine_cos_animated():
    X = np.linspace(0, 2*np.pi, 100)
    Y = np.sin(X)
    Z = np.cos(X)

    fig, ax = plt.subplots(2,1)
    ax[0].set_xlim([0, 2*np.pi])
    ax[0].set_ylim([-1.1, 1.1])
    sinegraph, = ax[0].plot([], [])
    dot, = ax[0].plot([], [], 'o', color='red')

    ax[1].set_xlim([0, 2*np.pi])
    ax[1].set_ylim([-1.1, 1.1])

    cosinegraph, = ax[1].plot([], [])
    dot_, = ax[1].plot([], [], 'o', color='b')

    def sine(i):
        sinegraph.set_data(X[:i],Y[:i])
        dot.set_data(X[i],Y[i])

        cosinegraph.set_data(X[:i],Z[:i])
        dot_.set_data(X[i],Z[i])

    anim = animation.FuncAnimation(fig, sine, frames=len(X), interval=50)
    plt.show()

def ohmslow():
    def rule2(c, m, A, dA):

        dQ = c*m*A**(m-1)*dA

        return dQ


    def rule3(dA,dB):
        
        #Need to use NumPy's sqrt so it knows what to do with
        #arrays!
        dQ = np.sqrt((dA)**2+(dB)**2)

        return dQ


    def rule4(c,m,n,A,dA,B,dB,C,dC,Q):

        dQ = Q*np.sqrt((m*dA/A)**2+(n*dB/B)**2)

        return dQ
    
    V = np.array([.243,.496,.998,1.995,2.998,4.007,4.967,6.004,7.01,8.01,9.01,10.02])   #Volts
    i = np.array([.003,.006,.012,.024,.036,.049,.061,.073,.086,.099,.112,.125])         #Amps
    
    errV = np.array([.001,.001,.001,.001,.001,.001,.001,.001,.001,.001,.001,.001])
    erri = np.array([.001,.001,.001,.001,.001,.001,.001,.001,.001,.001,.001,.001])
    
    #Re-assign variables as x, y, dy so that the following code may remain generic

    x = i     #this should be the array you want to plot on the x axis
    y = V
    dy = errV     #this should be your error in y array
    #Find the intercept and slope, b and m, from Python's polynomial fitting function

    b,m=np.polynomial.polynomial.polyfit(x,y,1,w=dy)
    #Write the equation for the best fit line based on the slope and intercept
    fit = b+m*x
    #Calculate the error in slope and intercept (you do not need to understand how these are calculated). def Delta(x, dy) is a function, and we will learn how to write our own at a later date. They are very useful!

    def Delta(x, dy):
        D = (sum(1/dy**2))*(sum(x**2/dy**2))-(sum(x/dy**2))**2
        return D

    D=Delta(x, dy)

    dm = np.sqrt(1/D*sum(1/dy**2)) #error in slope
    db = np.sqrt(1/D*sum(x**2/dy**2)) #error in intercept
    #Calculate the "goodness of fit" from the linear least squares fitting document

    def LLSFD2(x,y,dy):
        N = sum(((y-b-m*x)/dy)**2)
        return N
                        
    N = LLSFD2(x,y,dy)
    #Plot data on graph. Plot error bars and place values for slope, error in slope and goodness of fit on the plot using "annotate"

    plt.figure(figsize=(15,10))

    plt.plot(x, fit, color='green', linestyle='--')
    plt.scatter(x, y, color='blue', marker='o')


    #create labels  YOU NEED TO CHANGE THESE!!!
    plt.xlabel('Current (Amps)')
    plt.ylabel('Voltage (Volts)')
    plt.title('Ohms low')

    #plt.errorbar(x, y, yerr=dy, xerr=None, fmt=None) #don't need to plot x error bars

    plt.annotate('Slope ($m$/$s$) = {value:.{digits}E}'.format(value=m, digits=2),
                (0.05, 0.9), xycoords='axes fraction')

    plt.annotate('Error in Slope ($m$/$s$) = {value:.{digits}E}'.format(value=dm, digits=2),
                (0.05, 0.85), xycoords='axes fraction')

    plt.annotate('Goodness of fit = {value:.{digits}E}'.format(value=N, digits=2),
                (0.05, 0.80), xycoords='axes fraction')
    plt.show()

def sine_animated():
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = np.linspace(0, 2, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=200, interval=20, blit=True)
    plt.show()

def cosine_animated():
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = np.linspace(0, 2, 1000)
        y = np.cos(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=200, interval=20, blit=True)
    plt.show()

def magnitude_of_signal():
    # https://pythontic.com/visualization/signals/magnitude%20spectrum
    # Get time values of the signal
    time   = np.arange(0, 65, .25);
    # Get sample points for the discrete signal(which represents a continous signal)
    signalAmplitude   = np.sin(time)
    # plot the signal in time domain
    plt.subplot(211)
    plt.plot(time, signalAmplitude,'bs')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    # plot the signal in frequency domain
    plt.subplot(212)
    # sampling frequency = 4 - get a magnitude spectrum
    plt.magnitude_spectrum(signalAmplitude,Fs=4)
    # display the plots
    plt.show()

def frequesncy_domain():
    #plt.style.use('style/elegant.mplstyle')
    #The discrete1 Fourier transform (DFT) is a mathematical technique used to convert temporal or spatial data into frequency domain data. Frequency is a familiar concept, due to its colloquial occurrence in the English language: the lowest notes your headphones can rumble out are around 20 Hz, whereas middle C on a piano lies around 261.6 Hz; Hertz, or oscillations per second, in this case literally refers to the number of times per second at which the membrane inside the headphone moves to-and-fro. That, in turn, creates compressed pulses of air which, upon arrival at your eardrum, induces a vibration at the same frequency. So, if you take a simple periodic function, sin(10 × 2πt), you can view it as a wave:

    f = 10  # Frequency, in cycles per second, or Hertz
    f_s = 100  # Sampling rate, or number of measurements per second

    t = np.linspace(0, 2, 2 * f_s, endpoint=False)
    x = np.sin(f * 2 * np.pi * t)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(t, x)
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Signal amplitude');
    
    # Or you can equivalently think of it as a repeating signal of frequency 10 Hz (it repeats once every 1/10 seconds—a length of time we call its period). Although we naturally associate frequency with time, it can equally well be applied to space. For example, a photo of a textile patterns exhibits high spatial frequency, whereas the sky or other smooth objects have low spatial frequency.

    # Let us now examine our sinusoid through application of the DFT:
    
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * f_s

    ax[1].stem(freqs, np.abs(X))
    ax[1].set_xlabel('Frequency in Hertz [Hz]')
    ax[1].set_ylabel('Frequency Domain Magnitude')
    ax[1].set_xlim(-f_s / 2, f_s / 2)
    ax[1].set_ylim(-5, 110)
    plt.show()

def voice_signal():

    from scipy.io import wavfile

    rate, audio = wavfile.read('./data/voice.wav')
    #We convert to mono by averaging the left and right channels.

    audio = np.mean(audio, axis=1)
    # Then, we calculate the length of the snippet and plot the audio (Figure 4-2).
    N = audio.shape[0]
    L = N / rate
    #print(f'Audio length: {L:.2f} seconds')
    f, ax = plt.subplots()
    ax.plot(np.arange(N) / rate, audio)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [unknown]');
    plt.show()
def DFT():#Discrete fourier transform
    import time
    from scipy import fftpack
    from sympy import factorint
    K = 1000
    lengths = range(250, 260)
    # Calculate the smoothness for all input lengths
    smoothness = [max(factorint(i).keys()) for i in lengths]
    exec_times = []
    for i in lengths:
        z = np.random.random(i)
        # For each input length i, execute the FFT K times
        # and store the execution time
        times = []
        for k in range(K):
            tic = time.monotonic()
            fftpack.fft(z)
            toc = time.monotonic()
            times.append(toc - tic)
        # For each input length, remember the *minimum* execution time
        exec_times.append(min(times))
    f, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    ax0.stem(lengths, np.array(exec_times) * 10**6)
    ax0.set_ylabel('Execution time (µs)')

    ax1.stem(lengths, smoothness)
    ax1.set_ylabel('Smoothness of input length\n(lower is better)')
    ax1.set_xlabel('Length of input');
    plt.show()

def FT_rectangular():
    # Fourier transform of a rectangular pulse, we see significant side lobes in the spectrum:

    x = np.zeros(500)
    x[100:150] = 1

    X = fftpack.fft(x)

    f, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

    ax0.plot(x)
    ax0.set_ylim(-0.1, 1.1)

    ax1.plot(fftpack.fftshift(np.abs(X)))
    ax1.set_ylim(-5, 55);
    plt.show()

def kaiser_window():#https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch04.html
    # We can counter this effect by a process called windowing. The original function is multiplied with a window function such as the Kaiser window K(N, β). Here we visualize it for β ranging from 0 to 100:

    f, ax = plt.subplots()

    N = 10
    beta_max = 5
    colormap = plt.cm.plasma

    norm = plt.Normalize(vmin=0, vmax=beta_max)

    lines = [
        ax.plot(np.kaiser(100, beta), color=colormap(norm(beta)))
        for beta in np.linspace(0, beta_max, N)
        ]

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)

    sm._A = []

    plt.colorbar(sm).set_label(r'Kaiser $\beta$');
    plt.show()

def simple_palabola():
    x_cords = range(-50,50)
    y_cords = [x*x for x in x_cords]

    plt.plot(x_cords, y_cords)
    plt.show()

def electric_power_star_delta():
    # https://pyspice.fabrice-salvaire.fr/releases/v1.3/examples/ngspice-shared/external-source.html
    frequency = 50@u_Hz
    w = frequency.pulsation
    period = frequency.period

    rms_mono = 230
    amplitude_mono = rms_mono * math.sqrt(2)

    t = np.linspace(0, 3*float(period), 1000)
    L1 = amplitude_mono * np.cos(t*w)
    L2 = amplitude_mono * np.cos(t*w - 2*math.pi/3)
    L3 = amplitude_mono * np.cos(t*w - 4*math.pi/3)

    rms_tri = math.sqrt(3) * rms_mono
    amplitude_tri = rms_tri * math.sqrt(2)

    L12 = amplitude_tri * np.cos(t*w + math.pi/6)
    L23 = amplitude_tri * np.cos(t*w - math.pi/2)
    L31 = amplitude_tri * np.cos(t*w - 7*math.pi/6)

    figure = plt.figure(1, (20, 10))
    plt.plot(t, L1, t, L2, t, L3,
            t, L12, t, L23, t, L31,
            # t, L1-L2, t, L2-L3, t, L3-L1,
    )
    plt.grid()
    plt.title('Three-phase electric power: Y and Delta configurations (230V Mono/400V Tri 50Hz)')
    plt.legend(('L1-N', 'L2-N', 'L3-N',
                'L1-L2', 'L2-L3', 'L3-L1'),
            loc=(.7,.5))
    plt.xlabel('t [s]')
    plt.ylabel('[V]')
    plt.axhline(y=rms_mono, color='blue')
    plt.axhline(y=-rms_mono, color='blue')
    plt.axhline(y=rms_tri, color='blue')
    plt.axhline(y=-rms_tri, color='blue')
    plt.show()

def three_phase_wave():
        plt.title('three phase signal')

        ax = plt.subplot(111)
        t = np.arange(0.0, 5.0 , 0.01)
        s = np.sin(2*np.pi*t)
        ss = np.sin(2*np.pi*t+120)
        sss = np.sin(2*np.pi*t+240)
        line, = plt.plot(t,s, lw=2)
        line, = plt.plot(t,ss, lw=2)
        line, = plt.plot(t,sss, lw=2)
        plt.ylim(-3 ,3)
        plt.show()

def sinc_function():
        X = np.linspace(-6,6, 1024)
        Y = np.sinc(X)
        plt.title('sinc_function') # a little notation
        plt.xlabel('array variables') #adding xlabel
        plt.ylabel('random variables') #adding ylabel
        #plt.text(-5, 0.4, 'Matplotlib') # -5 is the x value and 0.4 is y value
        plt.plot(X,Y, color='r', marker ='o',markersize =3,markevery = 30, markerfacecolor='w',linewidth= 3.0,markeredgecolor = 'b')
        plt.show()

def distrubution_function():
        def gf(X, mu, sigma):
            a = 1. /(sigma*np.sqrt(2. * np.pi))
            b = -1. /(2. * sigma **2)
            return a * np.exp(b * (X - mu)**2)

        X = np.linspace(-6, 6, 1024)
        for i in range(64):
            samples = np.random.standard_normal(50)
            mu,sigma = np.mean(samples), np.std(samples)
            plt.plot(X, gf(X, mu, sigma),color = '.75',linewidth='.5')

        plt.plot(X,gf(X, 0., 1.),color ='.00',linewidth=3.)
        plt.show()

def area_area_under_curve():
        def func(x):
            return (x - 3) * (x - 5) * (x - 7) + 85


        a, b = 2, 9  # integral limits
        x = np.linspace(0, 10)
        y = func(x)

        fig, ax = plt.subplots()
        plt.plot(x, y, 'r', linewidth=2)
        plt.ylim(ymin=0)

        # Make the shaded region
        ix = np.linspace(a, b)
        iy = func(ix)
        verts = [(a, 0)] + list(zip(ix, iy)) + [(b, 0)]
        poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
        ax.add_patch(poly)

        plt.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
                 horizontalalignment='center', fontsize=20)

        plt.figtext(0.9, 0.05, '$x$')
        plt.figtext(0.1, 0.9, '$y$')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xticks((a, b))
        ax.set_xticklabels(('$a$', '$b$'))
        ax.set_yticks([])

        plt.show()

# area_area_under_curve()
# distrubution_function()
# sinc_function()
# three_phase_wave()
# electric_power_star_delta()
# simple_palabola()
# kaiser_window()
# FT_rectangular()
# DFT()
# voice_signal()
# frequesncy_domain()
# magnitude_of_signal()
# cosine_animated()
# sine_animated()
# sine_cos_animated()
# ohmslow()
# sine_cosine()   
# dumping_oscillation()
# pulse_generation()
# cosine_wave()
# sine_cosine()