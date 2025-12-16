# Bu kodda Genetik Algoritmanın fonksiyonlarını oluşturarak HW-4'e giriş yapıyorum.
import random
import matplotlib.pyplot as plt

def create_random_tour(n_nodes):
    """
    Rastgele geçerli bir TSP turu üretir.
    Başlangıç ve bitiş düğümü 0 olacak şekilde ayarlanır.
    """
    tour = list(range(1, n_nodes))
    random.shuffle(tour)
    return [0] + tour + [0]

def tour_length(tour, dist_matrix):

    length = 0.0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i]][tour[i+1]]
    return length

# Fitness hesaplama (maximizasyon olduğu için ters işlem yaptım)
def fitness(tour, dist_matrix):

    return 1.0 / tour_length(tour, dist_matrix)

# Popülasyon oluşturma
def initial_population(pop_size, n_nodes):

    return [create_random_tour(n_nodes) for _ in range(pop_size)]

# Selection
def tournament_selection(population, dist_matrix, k=3):

    contenders = random.sample(population, k)
    contenders.sort(key=lambda t: tour_length(t, dist_matrix))
    return contenders[0]

# Çaprazlama
def crossover(parent1, parent2):

    size = len(parent1) - 2
    a, b = sorted(random.sample(range(1, size + 1), 2))

    child = [None] * (size + 2)
    child[0] = child[-1] = 0
    child[a:b] = parent1[a:b]

    fill = [x for x in parent2 if x not in child]
    idx = 1
    for x in fill:
        while child[idx] is not None:
            idx += 1
        child[idx] = x

    return child

# Mutasyon Swap ile iki şehrin yerini değiştirme
def mutate(tour, mutation_rate=0.1):

    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(tour)-1), 2)
        tour[i], tour[j] = tour[j], tour[i]



