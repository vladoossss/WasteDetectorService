class TrackableObject:
    """ Небольшая обертка трекаемого обьекта """
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]

        self.counted = False