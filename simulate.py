#Code by Aryan Poonacha and Kushagra Ghosh
import numpy as np
from PIL import Image
from scipy.special import erf

from fluid import Fluid

import pandas as pd

def normalize(df):
    blacklist = ['Race', 'Year']
    result = df.copy()
    for feature_name in df.columns:
        if(feature_name in blacklist):
            continue
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (((df[feature_name] - min_value) / (max_value - min_value)) + 1)*2
    return result


data = pd.read_csv("datasets/data.csv", thousands=',')
data['n'] = data['n'].astype(int)
data['First'] = data['First'].astype(int)
data['Second'] = data['Second'].astype(int)
data['Third'] = data['Third'].astype(int)
data['Fourth'] = data['Fourth'].astype(int)
data['Fifth'] = data['Fifth'].astype(int)

normalized_data = normalize(data)

gb = data.groupby('Race')
groups = [gb.get_group(x) for x in gb.groups]
all = groups[0]
black = groups[1]

gb = normalized_data.groupby('Race')
groups = [gb.get_group(x) for x in gb.groups]
normalized_all = groups[0]
normalized_black = groups[1]

normalized_data["n"] = normalized_data["n"]*3


RESOLUTION = 300,300 #1200, 1200
DURATION = 1000

required_year = 1967
normalized_data = normalized_data[(normalized_data['Year'] >= 1967) & (normalized_data['Year'] <= 1977)]
print(normalized_data)
inflow_velocities = normalized_data.sort_values("Race")["First"].tolist()
inflow_radii = normalized_data.sort_values("Race")["n"].tolist()

INFLOW_PADDING = 1
INFLOW_DURATION = 10
INFLOW_RADIUS = 3
INFLOW_VELOCITY = 3
INFLOW_COUNT = 20 #data.shape[0] #number of years in dataset, 2020-1967

print('Generating fluid solver, this may take some time.')
fluid = Fluid(RESOLUTION, 'dye')

center = np.floor_divide(RESOLUTION, 2)
r = np.min(center) - INFLOW_PADDING

points = np.linspace(-np.pi, np.pi, INFLOW_COUNT, endpoint=False)
points = tuple(np.array((np.cos(p), np.sin(p))) for p in points)
normals = tuple(-p for p in points)
points = tuple(r * p + center for p in points)

inflow_velocity = np.zeros_like(fluid.velocity)
inflow_dye = np.zeros(fluid.shape)

i = 0
for p, n in zip(points, normals):
    print(i)
    mask = np.linalg.norm(fluid.indices - p[:, None, None], axis=0) <= inflow_radii[i]
    inflow_velocity[:, mask] += n[:, None] * inflow_velocities[i]
    i+=1
    inflow_dye[mask] = 1

frames = []
for f in range(DURATION):
    print(f'Computing frame {f + 1} of {DURATION}.')
    if f <= INFLOW_DURATION:
        fluid.velocity += inflow_velocity
        fluid.dye += inflow_dye

    curl = fluid.step()[1]
    # Using the error function to make the contrast a bit higher.
    # Any other sigmoid function e.g. smoothstep would work.
    curl = (erf(curl * 2) + 1) / 4

    color = np.dstack((curl, np.ones(fluid.shape), fluid.dye))
    color = (np.clip(color, 0, 1) * 255).astype('uint8')
    frames.append(Image.fromarray(color, mode='HSV').convert('RGB'))

print('Saving simulation result.')
frames[0].save('outputdecade1967.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)