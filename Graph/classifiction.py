
import pickle
import matplotlib.pyplot as plt


sizes = 1817,1816,1700,1900,1850,1830,
explode = (0.013, 0.013,0.013,0.013, 0.013,0.013,)
labels = [ 'FreshApples','FreshBananas','FreshOranges','RottenApples','RottenBanana','RottenOranges']
#autopact show percentage inside graph
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',)
plt.axis('equal')
plt.savefig('classification of dataset.jpg') # need to call before calling show
plt.show()