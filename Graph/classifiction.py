
import pickle
import matplotlib.pyplot as plt


sizes = 1157,1230,1013,
explode = (0.013, 0.013,0.013,)
labels = [ 'Borer','Choanephora','Sound',]
#autopact show percentage inside graph
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',)
plt.axis('equal')
plt.savefig('classification of dataset.jpg') # need to call before calling show
plt.show()