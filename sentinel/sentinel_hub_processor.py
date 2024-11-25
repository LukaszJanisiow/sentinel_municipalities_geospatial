from sentinelhub import (
    SHConfig,
    BBoxSplitter,
    CRS,
    BBox,
    bbox_to_dimensions,
    SentinelHubRequest,
    SentinelHubDownloadClient,
    DataCollection,
    MimeType,
    MosaickingOrder,
)
import geopandas as gpd
from math import ceil
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

class SentinelHubProcessor:
    def __init__(self, client_id, client_secret, shapefile_path):
        """Initialize the processor with SentinelHub credentials and shapefile."""
        self.config = SHConfig()
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        self.config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        self.shapefile = gpd.read_file(shapefile_path).to_crs(epsg=4326)
        self.aoi_bbox_list = []
        self.masked_images = []
        self.downloaded_data = []
        self.aoi_bbox = None
        self.area_of_interest = None
        self.aoi_split = None

    def define_area_of_interest(self, shapefile_index):
        """Define Area of Interest (AOI) from the shapefile."""
        self.area_of_interest = self.shapefile.loc[shapefile_index, "geometry"]
        if isinstance(self.area_of_interest, MultiPolygon):
            # Combine all polygons into a single list of exterior coordinates
            coords = []
            for polygon in self.area_of_interest.geoms:
                coords.extend(polygon.exterior.coords)
        elif isinstance(self.area_of_interest, Polygon):
            # Single Polygon: directly access its exterior
            coords = self.area_of_interest.exterior.coords
        else:
            raise ValueError("The geometry is not a Polygon or MultiPolygon.")
        # coords = self.area_of_interest.exterior.coords
        min_lat, min_lon, max_lat, max_lon = (
            min(c[1] for c in coords),
            min(c[0] for c in coords),
            max(c[1] for c in coords),
            max(c[0] for c in coords),
        )
        self.aoi_bbox = BBox([min_lon-0.01, min_lat-0.01, max_lon+0.01, max_lat+0.01], crs=CRS.WGS84)
        return self.aoi_bbox

    def split_aoi(self, resolution=10, max_tile_size=2400):
        """Split the AOI into smaller bounding boxes."""
        aoi_size = bbox_to_dimensions(self.aoi_bbox, resolution=resolution)
        cols = max(ceil(aoi_size[0] / max_tile_size), 1)
        rows = max(ceil(aoi_size[1] / max_tile_size), 1)
        splitter = BBoxSplitter([self.area_of_interest], CRS.WGS84, (cols, rows))
        self.aoi_split = splitter
        aoi_bigger = []
        for i in range(len(splitter.get_bbox_list())):
            bbox_list = [splitter.get_bbox_list()[i].lower_left[0] - 0.01, splitter.get_bbox_list()[i].lower_left[1] - 0.01, splitter.get_bbox_list()[i].upper_right[0] + 0.01, splitter.get_bbox_list()[i].upper_right[1] + 0.01]
            aoi_bbox = BBox(bbox=bbox_list, crs=CRS.WGS84)
            aoi_bigger.append(aoi_bbox)
        self.aoi_bbox_list = aoi_bigger
        return self.aoi_bbox_list

    def download_data(self, time_interval, evalscript = None, resolution=10, max_threads=5):
        """Download satellite data."""
        self.aoi_sizes = [
            bbox_to_dimensions(bbox, resolution=resolution) for bbox in self.aoi_bbox_list
        ]

        if evalscript is None:
            evalscript = """
                    //VERSION=3
                    function setup() {
                        return {
                            input: [{ bands: ["B02", "B03", "B04"] }],
                            output: { bands: 3 }
                        };
                    }
                    function evaluatePixel(sample) {
                        return [sample.B04, sample.B03, sample.B02];
                    }
                    """
        self.sh_requests = [
            SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            "s2l2a", service_url=self.config.sh_base_url
                        ),
                        time_interval=time_interval,
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=bbox,
                size=size,
                config=self.config,
            )
            for bbox, size in zip(self.aoi_bbox_list, self.aoi_sizes)
        ]

        download_client = SentinelHubDownloadClient(config=self.config)
        download_requests = [req.download_list[0] for req in self.sh_requests]
        self.downloaded_data = download_client.download(download_requests, max_threads=max_threads)
        return self.downloaded_data

    def create_masked_images(self, brightness = 1, resolution=10):
        """Create masked images from downloaded data."""
        self.masked_images = []
        for i, data in enumerate(self.downloaded_data):
            
            bbox = self.aoi_bbox_list[i]
            width, height = bbox_to_dimensions(bbox, resolution=resolution)
            img = Image.fromarray(data).convert("RGBA")

            # Convert geographic coordinates to pixel coordinates
            def geo_to_pixel(coords, bbox, img_width, img_height):
                min_x, min_y, max_x, max_y = bbox
                return [
                    ((x - min_x) / (max_x - min_x) * img_width,
                     (max_y - y) / (max_y - min_y) * img_height)
                    for x, y in coords
                ]
            # Create and apply a mask
            geometry = self.aoi_split.get_geometry_list()[i]
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            if isinstance(geometry, MultiPolygon):
                for polygon in geometry.geoms:
                    pixel_coords = (geo_to_pixel(polygon.exterior.coords, bbox, width, height))
                    draw.polygon(pixel_coords, fill=255)
            elif isinstance(geometry, Polygon):
                pixel_coords = geo_to_pixel(geometry.exterior.coords, bbox, width, height)
                draw.polygon(pixel_coords, fill=255)
            else:
                raise ValueError("Unsupported geometry type. Expected Polygon or MultiPolygon.")
            img_np = np.array(img)
            img_np[:, :, 3] = np.array(mask)
            if brightness > 1:
                enhancer = ImageEnhance.Brightness(Image.fromarray(img_np))  
                brighter_image = enhancer.enhance(4) 
                brighter_image_np = np.array(brighter_image)
                self.masked_images.append(brighter_image_np)
            elif brightness < 1:
                raise ValueError("Brightness must be equal or greater than 1.")
            else:
                self.masked_images.append(img_np)

        return self.masked_images

    def save_visible_patches(self, patch_size=64):
        """Extract and save visible patches from masked images."""
        patches = []
        for image in self.masked_images:
            img_np = np.array(image)
            for y in range(0, img_np.shape[0], patch_size):
                for x in range(0, img_np.shape[1], patch_size):
                    patch = img_np[y : y + patch_size, x : x + patch_size]
                    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                        continue
                    if np.sum(patch[:, :, 3] > 128) > 0:  # Visible pixels
                        patches.append(Image.fromarray(patch))
        return patches
