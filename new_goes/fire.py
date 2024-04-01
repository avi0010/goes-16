from datetime import datetime
from osgeo import ogr
import json

class Fire:
    """
    Represents a fire event.

    Attributes
    ----------
    id : int
        Identifier for the fire event.
    area_acre : float
        Area affected by the fire in acres.
    start_date : datetime
        Date and time when the fire was discovered.
    end_date : datetime
        Date and time when the fire was extinguished.
    geometry : ogr.Geometry
        Geometry object representing the spatial extent of the fire.
    """

    def __init__(
        self, id: int, area: float, start: datetime, end: datetime, geo: ogr.Geometry
    ) -> None:



        self.id = id
        self.area_acre = area
        self.start_date = start
        self.end_date = end
        self.geometry = geo

    @classmethod
    def parse(cls, fire):
        """
        Parse fire data and create Fire object.

        Parameters
        ----------
        fire_data : dict
            Dictionary containing fire data.

        Returns
        -------
        Fire
            A Fire object parsed from the provided data.
        """

        properties = fire["properties"]
        id = properties["poly_SourceOID"]
        area_acre = properties["poly_GISAcres"]

        fireDisoveryDateTime = properties["attr_FireDiscoveryDateTime"]

        fireControlDateTime = properties["attr_FireOutDateTime"]

        format_string = "%a, %d %b %Y %H:%M:%S %Z"
        start_date = datetime.strptime(fireDisoveryDateTime, format_string)
        end_date = datetime.strptime(fireControlDateTime, format_string)
        geometry = ogr.CreateGeometryFromJson(json.dumps(fire["geometry"]))
        return cls(id, area_acre, start_date, end_date, geometry)
