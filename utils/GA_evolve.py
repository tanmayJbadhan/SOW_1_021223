import numpy as np
import heapq

from typing import List, Tuple
#import mh.evaluate
import torchvision
import torch
from torchvision import transforms

train_dataset = torchvision.datasets.CIFAR10(
    root= './data', train = True,
    download =True, transform = None)
test_dataset  = torchvision.datasets.CIFAR10(
    root= './data', train = False,
    download =True, transform = None)
torch.manual_seed(1789)
train_dataset, _ = torch.utils.data.random_split(train_dataset, [0.01,0.99])
_, pirate_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
test_dataset, _ = torch.utils.data.random_split(test_dataset, [0.5, 0.5])
from .model_handler import ModelHandler
chenyaofo_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

mh = ModelHandler.download_model(
    "chenyaofo/pytorch-cifar-models", 'cifar10_resnet20',
    dataset = train_dataset,
    pirate_set = pirate_dataset,
    testset = test_dataset,
    transform = chenyaofo_transform,
    # device = 'cpu'
)

DIFFERENTIAL_WEIGHT = 0.7 # Arbitrary value for testing
KEY_LENGTH = 100 # Id
NUMBER_OF_GENERATION = 50 # Id
POPULATION_SIZE = 60 # Id
WIDTH, HEIGHT = 32, 32 # Image size
ALPHA = 0.1 # blending parameter for the watermark

F, K, G, N, W, H = DIFFERENTIAL_WEIGHT, KEY_LENGTH, NUMBER_OF_GENERATION, POPULATION_SIZE, WIDTH, HEIGHT # Aliases
TESTING = True
LOGGING = True



# Define the types
key_pixel = np.ndarray # size 5, type [int, int, int, int, int] # pixel RGB values in [0, 256[, pixel coordinates within max width and height
key = np.ndarray # an array of length K of pixels
logo_pixel = Tuple[int, int, int] # pixel RGB values in [0, 256[; the position is given by the pixel's indices in its logo
logo = Tuple[np.ndarray, int, int]  # The array is a matrix of pixel values; then the coordinates of the bottom-right corner which give the dimensions



### WITH LOGOS ###

def logo_fill(l: logo, new_cx: int, new_cy: int) -> logo:
    old_pixels, old_cx, old_cy = l
    # Create a new numpy array with the new dimensions, initialized to black pixels
    new_pixels = np.zeros((new_cy, new_cx, 3), dtype=int)

    # Calculate the range for copying over the pixel data
    copy_cx = min(old_cx, new_cx)
    copy_cy = min(old_cy, new_cy)

    # Copy over the pixel data from the old logo to the new logo
    new_pixels[:copy_cy, :copy_cx, :] = old_pixels[:copy_cy, :copy_cx, :]

    return new_pixels, new_cx, new_cy

def evolve_logo(logo1: logo, logo2: logo, logo3: logo, F: float) -> logo:
    # Determine the maximum dimensions
    max_cx = max(logo1[1], logo2[1], logo3[1])
    max_cy = max(logo1[2], logo2[2], logo3[2])

    # Resize all logos to the maximum dimensions
    logo1_pixels_resized, _, _ = logo_fill(logo1, max_cx, max_cy)
    logo2_pixels_resized, _, _ = logo_fill(logo2, max_cx, max_cy)
    logo3_pixels_resized, _, _ = logo_fill(logo3, max_cx, max_cy)

    # Perform the evolution calculation
    new_logo_pixels = logo1_pixels_resized + F * (logo2_pixels_resized - logo3_pixels_resized)
    
    # Clip to ensure pixel values are valid
    new_logo_pixels = np.clip(new_logo_pixels, 0, 255).astype(int)

    return new_logo_pixels, max_cx, max_cy

def watermarked_logo(img, l:logo):
    # Depends heavily on the structure of img
    print("placeholder watermarking")
    l_pixels_resized, _, _ = logo_fill(l, len(img), len(img[0]))
    return img + l_pixels_resized

def evaluate_logo(l:logo, f, dataset): # passing f and the dataset might not be necessary
    print("placeholder evaluate function")
    # def watermarking_function(img):
        # return watermarked_logo(img, l)
    # return mh.evaluate(watermarking_function)
    return 1.

def generate_population(N: int, max_width: int, max_height: int) -> List[logo]:
    return [
        (np.random.randint(0, 256, (np.random.randint(1, max_height + 1), 
                                    np.random.randint(1, max_width + 1), 3), dtype=int),
         np.random.randint(1, max_width + 1),
         np.random.randint(1, max_height + 1))
        for _ in range(N)
    ]


log = []


def differential_evolution_logo(D, f0, N:int=N, G:int=G, max_cx:int=10, max_cy:int=10, F:float=F, LOGGING:bool=LOGGING):
    # Randomly initialize population
    population = generate_population(N, max_cy, max_cx)
    fitness_scores = [evaluate_logo(l, f0, D) for l in population]
    best_key, best_fitness = max(zip(population, fitness_scores), key=lambda x: x[1])

    new_pop = [None for _ in range(N)]
    new_fitness_scores = [0 for _ in range(N)]
    # Log initial best key
    if LOGGING:
        global log 
        log = []
        log.append([best_key, best_fitness])

    # Iterate over G generations
    for _ in range(G - 1): # Already did one generation at random initialization
        for i in range(N):
            candidates = list(range(N))
            j, k, l = np.random.choice(candidates, 3, replace=False)
            
            new_logo = evolve_logo(population[j], population[k], population[l], F, K)
            new_fitness = evaluate_logo(new_logo, f0, D)

            if new_fitness > fitness_scores[i]:
                new_pop[i] = new_logo
                new_fitness_scores[i] = new_fitness
                if LOGGING and new_fitness > best_fitness:
                    best_key, best_fitness = new_logo, new_fitness
            else:
                new_pop[i] = population[i]  # Keep the old key if new one is not better

        population = new_pop  # Update the population for the next generation
        fitness_scores = new_fitness_scores
        # Log best candidate and its fitness
        if LOGGING:
            log.append([best_key, best_fitness])
            print(best_key, best_fitness)
    # Return the best candidate after G generations
    if LOGGING:
        return log
    else:
        return best_key, best_fitness
    




### NOW WITH KEYS ###

def distance(pixel1: key_pixel, pixel2: key_pixel) -> float: # The inputs are pixels: a doublet of coordinates.
    return np.linalg.norm(pixel1[3:]-pixel2[3:]) # Euclidean distance
    # return np.sum((pixel1[3:]-pixel2[3:])) # Manhattan distance

def pair(C1: key, C2: key, K:int=K) -> dict:
    heap = []  # empty heap
    pair_map = {}  # dictionary to map pixels from C1 to C2
    
    for i in range(K):
        for j in range(K):
            dist = distance(C1[i], C2[j])
            heapq.heappush(heap, (dist, i, j))

    while heap and len(pair_map) < K:
        _, i, j = heapq.heappop(heap)
        if i not in pair_map and j not in pair_map.values():
            pair_map[i] = j
    
    return pair_map

def reduce_overlap(pixels, W, H):
    # Create a 2D grid to track occupied positions
    occupancy_grid = np.zeros((W, H), dtype=bool)

    for pixel in pixels:
        x, y = int(pixel[3]), int(pixel[4])
        if occupancy_grid[x, y]:
            pixel[3], pixel[4] = find_new_position(x, y, occupancy_grid, W, H)
        occupancy_grid[int(pixel[3]), int(pixel[4])] = True

    return pixels

def find_new_position(x, y, occupancy_grid, W, H):
    # Spiral search for a new position
    dx, dy = 0, -1
    step_size = 0
    max_step = max(W, H)  # Maximum steps to prevent infinite loops

    for _ in range(max_step**2):
        if not (0 <= x < W and 0 <= y < H) or occupancy_grid[x, y]:
            # The position is outside bounds or occupied, so continue searching
            dx, dy = -dy, dx  # Change direction
            if dy == 0:
                step_size += 1  # Increase step size every full loop
        else:
            return x, y  # Found an unoccupied position

        x, y = x + dx, y + dy  # Take a step

    # If the loop completes without finding an unoccupied position, return the current position
    return x, y

def evolve_key(C1: key, C2: key, C3: key, F: float, K: int = K, W: int = W, H: int = H) -> key:
    pair_map_C1_C2 = pair(C1, C2, K)
    pair_map_C1_C3 = pair(C1, C3, K)

    # Extract the pairs based on the mapping
    C2_paired = C2[np.array(list(pair_map_C1_C2.values()))]
    C3_paired = C3[np.array(list(pair_map_C1_C3.values()))]

    # Calculate the new positions using the differential weight F
    new_pixels = C1 + F * (C2_paired - C3_paired)

    #overlapping_pixels = (C2_paired[:,3:] == C3_paired[:,3:]).sum(axis=-1) ==2 
    #new_pixels[overlapping_pixels, 3] = np.random.randint(0,H, size=overlapping_pixels.sum())
    #new_pixels[overlapping_pixels, 4] = np.random.randint(0,W, size=overlapping_pixels.sum())

    # For pixel values, the range is 0 to 255, for coordinates, it's 0 to W-1 or H-1 respectively
    new_pixels = np.clip(new_pixels, [0,0,0, 0, 0], [255,255,255, W - 1, H - 1])
    new_pixels = reduce_overlap(new_pixels, W, H)

    # Round and convert to appropriate data types
    new_pixels = np.rint(new_pixels).astype(int)  # Round and convert to int

    return new_pixels

def watermarked_key(img, k:key, alpha:float=ALPHA):
    # Depends heavily on the structure of img
    # print("placeholder watermarking")
    # print(img)
    for pixel in k:
        # print(pixel)
        R,G,B,x,y = pixel
        img[x,y] = img[x,y] + np.array([R,G,B], dtype=np.uint8)
        img[x,y] = np.clip(img[x,y], [0,0,0], [255,255,255])
    return img

def evaluate_key(k:key, f, dataset): # passing f and the dataset might not be necessary
    def watermarking_function(img):
        return watermarked_key(img, k)
    fitness =  mh.evaluate(watermarking_function)
    #print(fitnes)
    return fitness

def f0(img): 
    print("placeholder model")
    tag = np.random.randint(10)
    return tag
def differential_evolution_key(D, f0, N:int=N, G:int=G, K:int=K, F:float=F, LOGGING:bool=LOGGING):
    # Randomly initialize population
    population = [np.random.randint((256, 256, 256, W, H), size=(K, 5)) for _ in range(N)]
<<<<<<< HEAD
    print("first test")
    #evaluation_scores = [evaluate_key(key_n, f0, D) for key_n in population]
    #best_key, best_evaluation = max(zip(population, evaluation_scores), key=lambda x: x[1][0])
    #best_fitness, best_metrics = best_evaluation
    #fitness_scores = [e[0] for e in evaluation_scores]
    
    fitness_scores = [evaluate_key(key_n, f0, D) for key_n in population]
    best_key, best_fitness = max(zip(population, fitness_scores), key=lambda x: x[1])
    #best_fitness, best_metrics = best_evaluation
    #fitness_scores = [e[0] for e in evaluation_scores]
=======
    fitness_scores = [evaluate_key(k, f0, D) for k in population]
    best_key, best_evaluation = max(zip(population, fitness_scores), key=lambda x: x[1])
    best_fitness, best_metrics = best_evaluation
    
>>>>>>> 4c71a610e8684928b83814041f744288082f27f1
    new_pop = [None for _ in range(N)]
    #new_pop = [np.zeros((K, 5)) for _ in range(N)]
    new_fitness_scores = [0 for _ in range(N)]
    # Log initial best key
    if LOGGING:
        log = []
        log.append([best_key, best_fitness])
        print('initial fitness =', best_fitness)



    # Iterate over G generations
    for gen in range(1, G): # Already did one generation at random initialization
        for i in range(N):
            candidates = list(range(N))
            j, k, l = np.random.choice(candidates, 3, replace=False)
            new_key = evolve_key(population[j], population[k], population[l], F, K)
<<<<<<< HEAD
            new_fitness = evaluate_key(new_key, f0, D)
            # print(f'gen: {gen}, sample: {i}, tentative fitness: {new_fitness}')
=======
            new_fitness, new_metrics = evaluate_key(new_key, f0, D)

>>>>>>> 4c71a610e8684928b83814041f744288082f27f1
            if new_fitness > fitness_scores[i]:
                new_pop[i] = new_key
                new_fitness_scores[i] = new_fitness
                
                if LOGGING and new_fitness > best_fitness:
                    best_key, best_fitness, best_metrics = new_key, new_fitness, new_metrics
            else:
                new_pop[i] = population[i]  # Keep the old key if new one is not better

        population = new_pop  # Update the population for the next generation
        fitness_scores = new_fitness_scores
        # Log best candidate and its fitness
        if LOGGING:
<<<<<<< HEAD
            log.append([best_key, best_fitness])
            print(f'{gen}: fitness=', best_fitness)
=======
            log.append([best_key, best_fitness, best_metrics])
            print(f'{gen}: best_key=',best_key,'fitness=', best_fitness, 'metrics=', best_metrics)
>>>>>>> 4c71a610e8684928b83814041f744288082f27f1
    # Return the best candidate after G generations
    if LOGGING:
        return log
    else:
        return best_key, best_fitness, best_metrics



### TEST ZONE ###

def test_pair():
    K = 4  # Number of pixels
    # C1 and C2 have identical pixels
    C1 = np.array([[0,0,0, 0, 0], [0,0,0, 1, 1], [0,0,0, 2, 2], [0,0,0, 3, 3]])
    C2 = np.array([[0,0,0, 0, 0], [0,0,0, 1, 1], [0,0,0, 2, 2], [0,0,0, 3, 3]])
    
    # Expected: Each pixel should pair with itself
    expected_pairs = {0: 0, 1: 1, 2: 2, 3: 3}
    
    # Run the pairing function
    actual_pair_map = pair(C1, C2, K)
    
    # Check if the results match the expected pairs
    assert actual_pair_map == expected_pairs, f"Expected pairs do not match actual pairs: {actual_pair_map}"
    print("test_pair() passed.")

def test_pair2():
    K = 3  # Number of pixels
    C1 = np.array([[0,0,0, 0, 0], [0,0,0, 1, 1], [0,0,0, 2, 2]])
    C2 = np.array([[0,0,0, 1, 1], [0,0,0, 2, 2], [0,0,0, 3, 3]])
    
    # Expected pairing based on closest distance
    expected_pair_map = {0: 2, 1: 0, 2: 1}
    
    # Run the pairing function
    actual_pair_map = pair(C1, C2, K)
    
    # Check if the actual pairs match the expected pairs
    for i, expected_j in expected_pair_map.items():
        actual_j = actual_pair_map.get(i)
        assert actual_j == expected_j, f"Pixel {i} in C1 is expected to be paired with Pixel {expected_j} in C2, but got {actual_j if actual_j is not None else 'none'}"
    
    print("test_pair2() passed successfully.")

def test_evolve_key():
    K = 3  # Number of pixels
    F = 0.5  # Differential weight

    # Set up C1 and C2
    C1 = np.array([[0,0,0, 0, 0], [0,0,0, 10, 10], [0,0,0, 20, 20]])
    C2 = np.array([[0,0,0, 9, 9], [0,0,0, 19, 19], [0,0,0, 29, 29]])  # Specific distances from C1
    C3 = np.array([[0,0,0, 0, 0], [0,0,0, 0, 0], [0,0,0, 0, 0]])      # C3 will not contribute to the change

    # Expected new key
    # C1[0] moves halfway towards C2[2], C1[1] towards C2[0], and C1[2] towards C2[1]
    expected_new_key = np.array([[0,0,0, 14, 14], [0,0,0, 14, 14], [0,0,0, 29, 29]], dtype=int)

    # Run the evolve_key function
    actual_new_key = evolve_key(C1, C2, C3, F, K)

    # Check if the actual new key matches the expected new key
    assert np.allclose(actual_new_key, expected_new_key, 1), f"Expected new key {expected_new_key} does not match actual new key {actual_new_key}"
    print("test_evolve_key() passed successfully.")

def testdifferentialevolution():
    log1=differential_evolution_key(None, None)
    print(log1)


def all_test():
    test_pair()
    test_pair2()
    test_evolve_key()
    testdifferentialevolution()
    print("All tests completed.")

print("JJJ")
if __name__ == '__main__':
    if TESTING:
        all_test()
<<<<<<< HEAD
        print("JJJ")
=======
>>>>>>> 4c71a610e8684928b83814041f744288082f27f1
