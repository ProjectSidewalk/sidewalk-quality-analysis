from shapely.geometry import Point, MultiPolygon, Polygon
from rtree import index
import json


# Half the width/length of the bounding box of the point for r-tree
# Should be a small number
MIN_SIZE = 0.00001

class RegionStats:

    def __init__(self, geojson_path):
        """
        RegionStats object
        Initialization: RegionStats('data_seattle.geojson')
        """
        with open(geojson_path) as f:
            obj = json.load(f)

        self.feature_properties = list()
        self.geometries = list()
        for i in range(len(obj['features'])):
            for k in obj['features'][i]['geometry']['coordinates']:
                self.geometries.append(Polygon(k))
                self.feature_properties.append(obj['features'][i]['properties'])

        self.idx = index.Index()
        for i, polygon in enumerate(self.geometries):
            self.idx.insert(i, polygon.bounds)

    def get_properties(self, lng, lat):
        """
        Returns the properties for a longitude and latitude contained in the geojson
        
        Example:
        >>> get_properties(-122.309052, 47.598361)
        {'area': 0.753, 'density': 14689, 'population': 11055}

        """
        p = Point(lng, lat)
        box = (lng - MIN_SIZE, lat - MIN_SIZE, lng + MIN_SIZE, lat + MIN_SIZE)
        try:
            for i in self.idx.intersection(box):
                if self.geometries[i].contains(p):
                    return self.feature_properties[i]
        except Exception:
            return None


# if __name__ == '__main__':
#     r = RegionStats('data_seattle.geojson')
#     import pandas as pd
#     a = pd.read_csv('sidewalk-seattle-label_point.csv')
#     print(a.apply(lambda x: r.get_properties(x.lng, x.lat), axis=1))
