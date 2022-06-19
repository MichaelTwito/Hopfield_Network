import numpy as np
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time
from tempfile import TemporaryFile


start_time = time.time()
N = 100*100
P = 9
SIZE = (100,100)
NOISE_PERCENT = 10


def pic_to_array(pic_path,size=SIZE):
        pic_array = np.array(Image.open(pic_path).resize(size).convert('1')).flatten().astype(int)
        pic_array = np.where(pic_array==1, -1, pic_array)
        pic_array = np.where(pic_array==0, 1, pic_array)
        return pic_array

def arrays_to_pics_matrix(path,size=SIZE):
    counter = 0
    pics_matrix = np.empty([9,N], dtype=int)
    for pic in path:
        print (counter)
        pic_array = pic_to_array(pic)
        pics_matrix[counter] = pic_array
        counter+=1
    return pics_matrix

def build_weight_matrix(pics_matrix):
    w = np.zeros((N, N))
    # h = np.zeros((N))
    for i in range(N):
        print (i) 
        for j in range(N):
            for p in range(P):
                w[i, j] += (pics_matrix[p,i]*pics_matrix[p,j]).sum()
                if i==j:
                    w[i, j] = 0

    save_weight_matrix_to_file(w)
    return w

def set_random_sequence_of_update():
    sequnce_of_update = random.sample(range(0, N), N)
    return sequnce_of_update

def update(w,test_vector,test_pic_number,noise_percent,theta=0.5,time=5000):
    counter = 0
    random_sequence_update = set_random_sequence_of_update()
    for s in range(time):  
        epoch = s/N
        i = random_sequence_update[s % N] 
        u = np.dot(w[i][:],test_vector) - theta
        if u > 0:
            if test_vector[i] == -1:
                counter+=1
            test_vector[i] = 1
            
        elif u < 0:
            if test_vector[i] == 1:
                counter+=1
            test_vector[i] = -1

        if (epoch % 1 == 0):
            print ("Epoch: {}".format(epoch) + " Fixes: {}".format(counter) + " Test Pic Number: {}".format(test_pic_number) + " Noise Percent: {}".format(noise_percent))


    # print ("Number of changes: " + str(counter))
    return test_vector

def save_weight_matrix_to_file(weight_matrix):
    with open('weight50.npy', 'wb') as f:
        np.save(f, weight_matrix)

def add_noise(pic_path, percent=NOISE_PERCENT,size=SIZE):
    noise_percent = int(percent*N/100)
    pic_array = pic_to_array(pic_path)
    nosie_indexes = random.sample(range(0, N), noise_percent)
    for i in nosie_indexes:
        if pic_array[i] == 1:
            pic_array[i] = -1
        else:
            pic_array[i] = 1
    return pic_array

def build_images(pics_matrix, num_of_pics=P):
    images = []
    for i in range(num_of_pics):
        images.append(pics_matrix[i].reshape(SIZE))

    fig = plt.figure(figsize=(3, 3))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    for ax, im in zip(grid, images):
        ax.imshow(im)

def build_train_images_path(current_path):
    train_path = current_path+"/alphabet/"
    train_paths = []
    for i in sorted(os.listdir(train_path)):
        train_paths.append(train_path+i)
    return train_paths

def create_test_images(train_paths, test_dir_path, noise_percent= NOISE_PERCENT):
    for path in train_paths:
        blurred_test_vector = add_noise(path,noise_percent)
        blurred_test_vector = blurred_test_vector.reshape(SIZE)
        image = Image.fromarray(np.uint8(blurred_test_vector), mode = 'L').save(test_dir_path + path[-1], "png")

def build_test_images_path(current_path, number_of_tests=P):
    test_paths = []
    test_path = current_path+"/test/"
    for i in sorted(os.listdir(test_path)):
        test_paths.append(test_path+i)

    return test_paths

def run_epoch(weight_matrix,test_pics_matrix,noise_percent, number_of_tests=P):
    for i in range(number_of_tests):
        # print ("Test pic number: {}".format(i))
        test_pics_matrix[i] = update(weight_matrix, test_pics_matrix[i], i, noise_percent)
    return test_pics_matrix

def run_tests(train_paths,test_dir_path,current_path,weight_matrix,train_pics_matrix,number_of_tests):
    for noise_percent in [5,10,20,30,40,50]:
        for i in range(1,number_of_tests+1):
            create_test_images(train_paths, test_dir_path,noise_percent)
            test_paths = build_test_images_path(current_path)
            test_pics_matrix = arrays_to_pics_matrix(test_paths)
            build_images(test_pics_matrix)
            updated_test_pics_matrix = run_epoch(weight_matrix,test_pics_matrix, noise_percent)
            build_images(updated_test_pics_matrix)
            
    # print ("NOISSE :: " + str(noise_percent))
    return updated_test_pics_matrix

        
def main():

    current_path = os.getcwd()
    test_dir_path = current_path+"/test/"

    train_paths =  build_train_images_path(current_path)
    train_pics_matrix = arrays_to_pics_matrix(train_paths)    
    weight_matrix = build_weight_matrix(train_pics_matrix)
    # weight_matrix = np.load('100X100_neurons_examples/weight100X100.npy')
    # print (weight_matrix)
    updated_test_pics_matrix = run_tests(train_paths,test_dir_path,current_path,weight_matrix,train_pics_matrix,1)

    build_images(train_pics_matrix)

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.show()

if __name__ == "__main__":
    main()