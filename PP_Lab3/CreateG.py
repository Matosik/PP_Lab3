import matplotlib.pyplot as plt

def parse_results(filename):
    sizes = []
    times = []
    with open(filename, 'r') as file:
        for line in file:
            size = int(line.split('size')[1].split('x')[0])
            time = int(line.split('-')[1].strip().split('ms')[0].strip())
            sizes.append(size)
            times.append(time)
    return sizes, times

def plot_results(*data, labels):
    plt.figure(figsize=(10, 5))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']  # Расширяем список цветов
    markers = ['o', 's', '^', 'd', 'x', '+', '>', '<', 'p', '*']  # Разные маркеры для различия линий

    for i, (sizes, times) in enumerate(data):
        color = colors[i % len(colors)]  # Циклическое использование цветов
        marker = markers[i % len(markers)]  # Циклическое использование маркеров
        plt.plot(sizes, times, marker=marker, color=color, label=labels[i])

    plt.title('Зависимость времени вычисления от размера матрицы (MPI) Супер-пупер Королев')
    plt.xlabel('Размер матрицы (n x n)')
    plt.ylabel('Время (мс)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Пример использования
filenames = ['dataKorolev/ResultExperimentMPI_1thread.txt', 'dataKorolev/ResultExperimentMPI_2thread.txt','dataKorolev/ResultExperimentMPI_4thread.txt','dataKorolev/ResultExperimentMPI_8thread.txt']
data = [parse_results(filename) for filename in filenames]
labels = ['1 поток', '2 потока',"4 потока", "8 потоков"]

plot_results(*data, labels=labels)