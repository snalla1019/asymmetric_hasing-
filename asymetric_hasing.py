# asymmetric hashing
class AH:

# contructor
def __init__(self, data, num_partitions, num_centroids):
self.X = data
self.I, self.J = self.X.shape
self.L = num_partitions
self.M = num_centroids
# get feature partitions
self.partitions = self._get_partitions()
# to be initialized
self.hashed_X = None
self.centroids = None

# function to determine the partition split points
def _get_partitions(self):
incr = math.floor(self.J / self.L)
# initialize the array of partition indexes
out = []
lb = 0
for i in range(0, self.L):
# need to handle the last partition separately
ub = lb + incr if i < self.L - 1 else self.J
out.append(range(lb, ub))
lb = ub
# return the partition indexes
return out

# function to run kmeans for each partitions
def hashing(self):
self.hashed_X = np.zeros((self.I, self.L)) # I x L
self.centroids = [] # L x M
# for each partition
for l in range(0, self.L):
# retrieve partition
partition = self.partitions[l]
# run kmeans
kmeans = KMeans(n_clusters=self.M).fit(self.X[:, partition])
# get the assigned centroid for each data point
self.hashed_X[:, l] = kmeans.labels_
# get the centroids
self.centroids.append(kmeans.cluster_centers_)
# debug print
print('Finished running kmeans for partition: ' + str(partition))

# compute distances over all partitions and centroids for the target x
def _compute_distances(self, x):
distances = np.zeros((self.L, self.M))
for l in range(0, self.L):
# extract the vector of partial features
xl = x[self.partitions[l]]
for m in range(0, self.M):
centroid = self.centroids[l][m]
# compute squared 2-norm between the centroid
# and the partial vector
distances[l][m] = math.pow(np.linalg.norm(xl - centroid), 2.0)
return distances

# function to find k nearest neighbors to the target x
def find_neighbors(self, x, top_k):
hashed_distances = self._compute_distances(x)
distances = np.zeros(self.I)
for i in range(0, self.I):
hashed_x = self.hashed_X[i, :]
for l in range(0, self.L):
distances[i] += hashed_distances[l, int(hashed_x[l])]
distances[i] = math.sqrt(distances[i])
# find k nearest neighbors
idx = np.argpartition(distances, top_k)[:top_k]
idx.sort()
# return the neighbor indexes and the approximated distances
return [idx, distances[idx]]

@staticmethod
def test():
# run AH
ah = AH(Data.X, num_partitions=5, num_centroids=3)
ah.hashing()
# get top k results
idx, distances = ah.find_neighbors(Data.target, Data.top_k)
print('AH: ' + str(Data.top_k) + ' nearest neighbors: ' + str(idx))


# locality-sensitive hashing
class LH:

# contructor
def __init__(self, data, num_planes):
self.I, self.J = data.shape
self.X = data
self.K = num_planes
self.planes = np.random.random((self.K, self.J))
self.assignments = None

# function to assign the given x to the cloest plane
def _get_plane(self, x):
return np.argmin(np.array([
np.linalg.norm(x - self.planes[k, :]) for k in range(0, self.K)
]))

# function to assign each base x to the cloest plane
def hashing(self):
self.assignments = np.array([
self._get_plane(self.X[i, :]) for i in range(0, self.I)
])

# function to find k nearest neighbors to the target x
def find_neighbors(self, x, top_k):
# find the closest plane
idxs = np.where(self.assignments == self._get_plane(x))[0]
distances = np.array([
np.linalg.norm(self.X[idxs[i], :] - x) for i in range(0, len(idxs))
])
# function to find k nearest neighbors to the target x
idx = np.argpartition(distances, top_k)[:top_k]
idx.sort()
return [idxs[idx], distances[idx]]

@staticmethod
def test():
# run LH
lh = LH(Data.X, num_planes=5)
lh.hashing()
# get top k results
idx, distances = lh.find_neighbors(Data.target, Data.top_k)
print('LH: ' + str(Data.top_k) + ' nearest neighbors: ' + str(idx))


if __name__ == '__main__':
AH.test()
LH.test()