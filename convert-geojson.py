#%%
import json

#%%
with open('data_seattle.json') as f:
    obj = json.load(f)

#%%
geojson_obj = {
    'type': 'FeatureCollection',
    'features': []
}

#%%
for name in obj:
    geometry = obj[name]['geometry']
    features = {
        'area': obj[name]['area'],
        'population': obj[name]['population'],
        'density': obj[name]['density']
    }

    geojson_obj['features'].append({
        'type': 'Feature',
        'geometry': geometry,
        'properties': features
    })

#%%
with open('data_seattle.geojson', 'w') as f:
    json.dump(geojson_obj, f, indent=1, sort_keys=True)

#%%
