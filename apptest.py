import app
from keras import models

def retSums():
    model = models.load_model('model6.h5')
    sums = []
    for n in range(0, 10):
        sum = app.playVideo("proj-lvl3-data/video-" + str(n) + ".avi", model)
        sums.append(sum)
    return sums