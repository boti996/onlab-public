import pickle
import numpy as np
import gc

import helpers as helper


# TODO: uint8 beállítása
# Serialize image files -- no subdivision with /255
def main():
    images = pickle.load(open("datas/images_mlnd/image/full_CNN_train.p", "rb"))
    size = (320, 256)
    file = open('datas/images_mlnd/image/my_coeff_images.p', "wb")
    no_images = len(images)
    pack_size = 1000
    no_rounds = no_images // pack_size
    no_rounds += 1
    print("Number of rounds: "+str(no_rounds))
    for n in range(0, no_rounds):
        pack = images[n*pack_size:(n+1)*pack_size]
        pack = helper.resize_images(pack, size)
        # pack = np.array(pack) / 255
        pickle.dump(pack, file)
        gc.collect()
        print(str(n)+"/"+str(no_rounds))
    file.close()


main()
