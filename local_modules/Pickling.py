import pickle
def pickle_item(filepath, object):
    filehandler = open(filepath,"wb")
    pickle.dump(object, filehandler)
    filehandler.close()
    print("Pickling completed") 