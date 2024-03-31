import numpy as np
import matplotlib.pyplot as plt
import csv

def haarMatrix(n, normalized=False):
    # allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h

def visualize_simple(S, S_new, title, subtitle):
    plt.figure(figsize=(10, 5))
    plt.plot(S, label='S', marker='o', linestyle='-')
    plt.plot(S_new, label='S_DWT', marker='x', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(subtitle, fontsize=12)
    plt.suptitle(title, fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize(S, S_new, Sfou, title, subtitle):
    plt.figure(figsize=(10, 5))
    plt.plot(S, label='S', marker='o', linestyle='-')
    plt.plot(S_new, label='S_DWT', marker='x', linestyle='--')
    plt.plot(np.fft.ifft(Sfou), label='S_Fourier', marker='x', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(subtitle, fontsize=12)
    plt.suptitle(title, fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.show()

def dft(S,frac):
    f = np.fft.fft(S)
    n = len(S)
    # energy associated with each coefficient
    E = []
    for i in range(0, len(f)):
        E.append(np.real(f[i])**2+np.imag(f[i])**2)
    Etot = np.sum(E)
    num = int(frac*n) # number of indexes to keep
    indeces = np.argsort(E)[-num:] # argmax
    Sfou = np.zeros_like(f)
    Erel = 0
    for i in indeces:
        Sfou[i] = f[i]
        Erel += E[i]

    rf = Erel/Etot

    return Sfou, rf
    

def dwt(S, frac):
    n = len(S)
    W = haarMatrix(n)
    a = []
    Sapp = []
    ren = 0
    # computation of the weights
    a.append(np.mean(S)) # psi(1,1)
    # how many: 2^{k-1} weights of order k, for k = 1,...,log_2 (n)
    for k in range(1,int(np.log2(n))+2): # k is the order of the weight
        for i in range(1,2**(k-1),2):
            # indexes for extract the subsequence from S
            start1 = int(((i-1)*n/(2**(k-1)))+1)
            end1 = int((i*n)/(2**(k-1)))
            subs1 = S[start1-1:end1]
            start2 = int(((i)*n/(2**(k-1)))+1)
            end2 = int(((i+1)*n)/(2**(k-1)))
            subs2 = S[start2-1:end2]
            phi1 = np.mean(subs1)
            phi2 = np.mean(subs2)
            a.append((phi1-phi2)/2)

    if frac == 0:
        return a, W

    # energy associated with each weight
    E = []
    for i in range(0, len(a)):
        E.append((a[i]*np.linalg.norm(W[i]))**2)
    Etot = np.sum(E)
    num = int(frac*n) # number of indexes to keep
    indeces = np.argsort(E)[-num:] # argmax
    ared = np.zeros_like(a)
    Erel = 0
    for i in indeces:
        ared[i] = a[i]
        Erel += E[i]

    ren = Erel/Etot

    Sapp = np.dot(np.transpose(ared),W)

    return ared, W, Sapp, ren

def open_csv(name):
    with open(name, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        data = []
        numbers = []
        for row in csv_reader:
            data.append(row)
            numbers.append(float(row[-1]))
    return data, numbers

def cut_to_nearest_power_of_2(array):
    array_length = len(array)
    log_length = np.log2(array_length)
    nearest_power_of_2 = int(log_length)
    new_length = 2 ** nearest_power_of_2
    return array[:new_length]

def test():
    # testing with data of exercise sheet #5
    S = [2, 6, 6, 3, 3, 3, 5, 5, 3, 4, 6, 3, 2, 3, 6, 4]
    a, W = dwt(S,0)
    print(a)
    print(W)

    ared, W, Sapp, ren = dwt(S,1/4)
    print(ared)
    print(W)
    print(Sapp)
    print(ren)

def display(csv, title, frac):
    _, air_temperature = open_csv(csv)
    S = cut_to_nearest_power_of_2(air_temperature)
    ared, W, Sapp, ren = dwt(S,frac)
    Sfou, rf = dft(S,frac)
    visualize_simple(S, Sapp, title = title, 
            subtitle=('Information retained = ', round(ren*100,2),'%, Chosen fraction of weights = ', frac))
    visualize(S, Sapp, Sfou, title = title, 
            subtitle=('Information retained = ', round(ren*100,2),'%, Chosen fraction of weights = ', frac, ', Information retained (Fourier) = ',round(rf*100,2),'%'))
    # print(ared)
    # print(W)

def main():

    # test()   

    display("min_temp.csv", "Minimum Temperature in Kuopio (hourly), 1.1.2024 - 4.1.2024", 0.15)
    display("snow_depth.csv", "Snow Depth in Kuopio (daily), 1.11.2023 - 1.2.2024", 0.1)
    display("gust.csv", "Maximum Gust Speed in Kuopio (hourly), 1.1.2024 - 4.1.2024", 0.2)
    display("prec.csv", "Monthly Precipitation in Kuopio, 1.1.2020 - 1.1.2024", 0.1)
    display("humidity.csv", "Average Realtive Humidity in Kuopio (hourly), 5.2.2024 - 8.2.2024", 0.1)


if __name__=="__main__": 
    main() 