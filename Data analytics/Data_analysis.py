import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\mariu\PycharmProjects\machineLearning\train.csv')
print(df.info())


# Analysis of gross profits by sex

words = df.iloc[:, [0, 7]]
e = []
for i in range(len(words)):
    if words.iloc[i,1] > words.iloc[i,0] :
        e.append(i)

malespeak= np.array(e)

malebool = df.index.isin(malespeak)
femalebool = ~malebool

maledf = df.iloc[malebool]
femaledf = df.iloc[femalebool]
malegross = maledf[['Gross']]
femalegross = femaledf[['Gross']]

print(malegross.describe())
print(femalegross.describe())

# Analysis of nr of speaking roles in movies by sex
speaking_roles = df[['Number of male actors', 'Number of female actors']]

roles_array = speaking_roles.to_numpy()

total_roles = np.sum(roles_array, axis=0)
print(total_roles)

y = np.array([total_roles[0], total_roles[1]])
roles = np.array(['Male', 'Female'])
plt.figure(1)

plt.bar(roles, y)
plt.xlabel('sex')
plt.ylabel('Speaking roles')
plt.title('Number of actors with major speaking roles for each sex, summed up over 1039 movies')

plt.show()

