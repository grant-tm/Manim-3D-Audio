from manim import *
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, rfft

window_size = 256

sample_rate, samples = wavfile.read("Daniel Caesar - Let Me Go (Instrumental).wav")
#samples = samples[:,0] + samples[:,1]
#samples = np.convolve(samples, np.ones(10)/10, mode='same')
#samples = (samples - min(samples)) / (max(samples) - min(samples))

def split_windows(a, size):
    return np.split(a, np.arange(size,len(a),size))

sample_windows = split_windows(samples, window_size)

cur_window = 0
grid_x = 64
grid_y = 16

# GRID MEMORY
grid_fft = np.array([np.zeros(int(window_size/2)) for i in range(0, grid_x)])

def increment_window(time):
    global cur_window
    if(np.floor(time) > cur_window):
        cur_window = int(np.floor(time))
        cycle_fft(cur_window)
    return

def perform_fft(window):
    if(window < 0):
        return np.zeros(int(window_size/2))
    result = rfft(sample_windows[window]).real[:int(window_size/2)]
    #result = np.log1p(abs(result))
    #result = result / 4
    result = (result - np.amin(result)) / (np.amax(result) - np.amin(result))
    result = np.convolve(result, np.ones(10)/10, mode='same')

    return result

def cycle_fft(window):
    if(window % 1 != 0):
        return
    window = int(window)
    
    global grid_fft
    grid_fft = grid_fft[:-1]
    grid_fft = np.insert(grid_fft, 0, perform_fft(window), axis=0)
    return

def linear_interpolation(x, y, window):
    
    #x = abs(x)
    y = abs(y)
    
    increment_window(window)
    
    # get the x and y values of the nearest grid points
    x_bot = int(np.floor(x))
    x_top = int(np.ceil(x))    
    y_bot = int(np.floor(y))
    y_top = int(np.ceil(y))
    
    # calculate weights for interpolation
    x_bot_weight = abs(x_top - x)
    x_top_weight = abs(x_bot - x)
    y_bot_weight = abs(y_top - y)
    y_top_weight = abs(y_bot - y)
    
    # no interpolation needed
    if(x%1 == 0 and y%1 == 0):
        return grid_fft[int(x), int(y)]
    
    # linear interpolation in y
    elif(x%1 == 0):
        z1 = grid_fft[int(x), y_bot] * y_bot_weight
        z2 = grid_fft[int(x), y_top] * y_top_weight
        return z1 + z2
    
    # linear interpolation in x
    elif(y%1 == 0):
        z1 = grid_fft[x_bot, int(y)] * x_bot_weight
        z2 = grid_fft[x_top, int(y)] * x_top_weight
        return z1 + z2
        
    # bilinear interpolation
    else:
        z1 =  grid_fft[x_bot, y_bot] * y_bot_weight
        z1 += grid_fft[x_bot, y_top] * y_top_weight
        z1 *= x_bot_weight
        
        z2 =  grid_fft[x_top, y_bot] * y_bot_weight
        z2 += grid_fft[x_top, y_top] * y_top_weight
        z2 *= x_top_weight
        return z1 + z2

def cosine_interpolation_1d(a, b, x):
    ft = x * np.pi
    f = (1 - np.cos(ft)) * 0.5
    return b * (1 - f) + a * f

def cosine_interpolation(x, y, window):
    
    #x = np.around(x, 3)
    #y = np.around(y, 3)
    increment_window(window)
    
    # get the x and y values of the nearest grid points
    x_bot = int(np.floor(x))
    x_top = int(np.ceil(x))    
    y_bot = int(np.floor(y))
    y_top = int(np.ceil(y))
    
    if(x%1 == 0 and y%1 == 0):
        return grid_fft[int(x), int(y)]
    elif(x%1 == 0):
        return cosine_interpolation_1d(grid_fft[int(x), y_bot], grid_fft[int(x), y_top], y_top-y)
    elif(y%1 == 0):
        return cosine_interpolation_1d(grid_fft[x_bot, int(y)], grid_fft[x_top, int(y)], x_top-x)
    else:
        z1 = cosine_interpolation_1d(grid_fft[x_bot, y_bot], grid_fft[x_bot, y_top], y_top-y)
        z2 = cosine_interpolation_1d(grid_fft[x_top, y_bot], grid_fft[x_top, y_top], y_top-y)
        return cosine_interpolation_1d(z1, z2, x_top-x)
    
# for i in range(0, 10):
    
#     arr = np.array([[cosine_interpolation(x/10, y/10, i) for x in range(0, (grid_x-1)*10)] for y in range(0, (grid_y-1)*10)])
#     print(" ")
#     print(i)
#     print(arr) 

class plane(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        
        number_plane = NumberPlane(
            x_range=(-0.01, grid_x + 0.01, (grid_x / 8)),
            y_range=(-0.01, grid_y + 0.01, (grid_y / 4)),
            x_length=10,
            y_length=5,
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 2,
                "stroke_opacity": 1
            }
        )
        
        self.set_camera_orientation(phi=65 * DEGREES, theta=0 * DEGREES)
        phi, theta, focal_distance, gamma, distance_to_origin = self.camera.get_value_trackers()
        
        dist = distance_to_origin.get_value()
        distance_to_origin.set_value(0.5)
        phi.set_value(0 * DEGREES)
        
        self.play(
            phi.animate.set_value(65 * DEGREES),
            distance_to_origin.animate.set_value(dist),
            theta.animate.set_value(-210 * DEGREES),
            Write(number_plane),
            run_time=4,
            rate_func = rush_from
        )
            
        start_window = 0
        num_windows = 10000
        tracker = ValueTracker(start_window)
        
        number_plane.prepare_for_nonlinear_transform()
        always_redraw(
            lambda: number_plane.apply_function(
                lambda p: np.array([
                    p[0],
                    p[1], 
                    cosine_interpolation(p[0], p[1], tracker.get_value())
                ])
            )
        )
        
        self.play( 
            tracker.animate.set_value(num_windows),
            theta.animate.set_value(-640 * DEGREES),
            run_time = num_windows * window_size / sample_rate,
            rate_func = linear
        )
        self.wait()