from shapely.geometry import Polygon, Point
import json

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
            self.feature_properties.append(obj['features'][i]['properties'])
            assert len(obj['features'][i]['geometry']['coordinates']) == 1
            self.geometries.append(Polygon(obj['features'][i]['geometry']['coordinates'][0]))
    
    def get_properties(self, lng, lat):
        """
        Returns the properties for a longitude and latitude contained in the geojson
        
        Example:
        >>> get_properties(-122.309052, 47.598361)
        {'area': 0.753, 'density': 14689, 'population': 11055}

        """
        p = Point(lng, lat)
        # TODO use r-tree
        for i in range(len(self.geometries)):
            if self.geometries[i].contains(p):
                return self.feature_properties[i]
