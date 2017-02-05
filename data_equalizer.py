import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import math

csv_file = "driving_log.csv"
csv_file_out="new_driving_log.csv"
df = pd.read_csv(csv_file, header=None)

angles = df[3]
destribution = Counter(angles)

mean = np.mean(list(destribution.values()))
mean = math.floor(mean)

print("Totaal different angles: " + str(len(destribution)))
print("Mean amount of examples/angle: " + str(mean))

#Equalize data
final_df = pd.DataFrame()
for key in destribution:
	partition = df.loc[angles == key]
	diff = mean - len(partition)
	# to many samples
	if (diff <= 0):
		sample = partition.sample(n=mean)
		final_df = pd.concat([final_df,sample])
	# to few samples
	elif (diff > 0):
		for i in range(1, diff+1):
			sample = partition.sample(n=1)
			final_df = pd.concat([final_df,sample])


# write new data destribution to file
final_df.to_csv(csv_file_out, header=False, index=False, encoding='utf-8')


# Final destribution values for debuging
final_angles = final_df[3]
final_destribution = Counter(final_angles)

final_mean = np.mean(list(final_destribution.values()))
final_mean = math.floor(final_mean)

print("Totaal different angles after equalization: " + str(len(final_destribution)))
print("Mean amount of examples/angle after equalization: " + str(final_mean))

final_df[3] = final_df[3].apply(pd.to_numeric)
final_df.hist(column=3, bins = final_mean)
plt.show()

