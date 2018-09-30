import numpy
import sys
import random
import numpy as np

class Movie:
    def __init__(self, title="", year=0, runtime=0):
        self.title = title
        self.year = year
        self.runtime = runtime
        if runtime < 0:
            self.runtime = 0
    
    def __repr__(self):
        return "%s (%s) - %s mins" % (self.title, self.year, self.runtime)

    def hourmin(self):
        hr = self.runtime // 60
        min = self.runtime % 60
        return hr, min

def create_movie_list():
    list = []
    list.append(Movie("Jurassic World", 2017, 124))
    list.append(Movie("Hello World", 2017, 164))
    list.append(Movie("KuaiLe de JiaTing", 2017, 134))
    list.append(Movie("Captain America", 2017, 126))
    list.append(Movie("Avatar", 2017, 179))
    return list

def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    # random = random.random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100
        
        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array



def main(argv):
    # print("  *** Full List *** ")
    # list = create_movie_list()
    # for m in list:
    #     print(m)

    # print("  *** Filtered List *** ")
    # # newlist = [x.append(): for y in list: x.append(y)]
    # newlist = [item for item in list if item.runtime > 150]
    # for m in newlist:
    #     print(m)

    # ratings = {}
    # for m in list:
    #     ratings[m.title] = random.random() * 5
    
    # for k, v in ratings.items():
    #     print("{} - {:.02f}".format(k, v))
    data = get_movie_data()
    print(data.shape[0])
    print(data.shape[1])

    head = data[:2]
    print(head)
    columns = data[:,[1,2]]
    print(columns)
    columns = data.T[1:]

    

    print(data.T[1])
    


    
    
    




if __name__ == "__main__":
    main(sys.argv)
