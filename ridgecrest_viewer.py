from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapefile
import vtk


class RidgecrestVTKViewer:
    def __init__(self, csv_path: Path, dem_path: Path, faults_path: Path | None = None):
        #i load the three main datasets up front. the earthquakes, terrain, and optional faults
        self.df = self._load_catalog(csv_path)
        self.dem, self.bounds = self._load_dem(dem_path)

        #i treat the largest event in the filtered catalog as the mainshock
        self.mainshock = self.df.loc[self.df["mag"].idxmax()].copy()
        self.max_days = float(self.df["days_since_start"].max())
        self.max_depth = float(self.df["depth"].max())

        self.faults_path = faults_path
        self.has_fault_data = faults_path is not None and Path(faults_path).exists()

        self.recent_window_days = 1.5

        #these vertical offsets are mostly for readability so the layers do not fight visually
        self.z_scale = 0.00012
        self.point_lift = 0.020
        self.contour_lift = 0.035
        self.fault_lift = 0.060

        self.depth_mode = False
        self.depth_z_scale = 0.0035
        self.depth_base_offset = 0.010
        self.depth_terrain_opacity = 0.35

        self.show_contours = True
        self.remove_old_shocks = False
        self.show_faults = self.has_fault_data

        #older aftershocks fade out so the screen does not turn into one giant purple blob
        self.old_far_opacity = 0.20
        self.old_mid_opacity = 0.30
        self.old_near_opacity = 0.45
        self.recent_point_opacity = 1.00

        self.playing = False
        self.current_day = 0.0
        self.play_step_days = max(self.max_days / 220.0, 0.08)
        self.timer_ms = 30

        #these contours are only meant to show location-density clustering
        #they dont use magnitude, depth, shaking intensity, or stress
        self.contour_min_events = 8
        self.contour_halo_radius = 0.0022
        self.contour_color_radius = 0.0014
        self.contour_bins = 120
        self.contour_sigma = 3.2
        self.contour_levels = 8
        self.contour_scalar_min = 0.10
        self.contour_scalar_max = 1.0

        self.fault_halo_radius = 0.0030
        self.fault_color_radius = 0.0018
        self.fault_min_segment_length = 0.0008

        self.renderer = None
        self.render_window = None
        self.interactor = None

        self.terrain_grid = None
        self.terrain_actor = None

        self.subsurface_polydata = vtk.vtkPolyData()
        self.subsurface_actor = None
        self.subsurface_outline_actor = None

        self.old_far_polydata = vtk.vtkPolyData()
        self.old_mid_polydata = vtk.vtkPolyData()
        self.old_near_polydata = vtk.vtkPolyData()
        self.recent_polydata = vtk.vtkPolyData()

        self.old_far_mapper = None
        self.old_mid_mapper = None
        self.old_near_mapper = None
        self.recent_mapper = None

        self.old_far_actor = None
        self.old_mid_actor = None
        self.old_near_actor = None
        self.recent_actor = None

        self.mainshock_actor = None

        self.contour_polydata = vtk.vtkPolyData()
        self.contour_halo_tube = None
        self.contour_color_tube = None
        self.contour_halo_actor = None
        self.contour_color_mapper = None
        self.contour_color_actor = None

        self.fault_polydata = vtk.vtkPolyData()
        self.fault_halo_tube = None
        self.fault_color_tube = None
        self.fault_halo_actor = None
        self.fault_color_actor = None

        self.fault_curtain_polydata = vtk.vtkPolyData()
        self.fault_curtain_actor = None
        self.fault_curtain_outline_actor = None

        self.title_actor = None
        self.info_actor = None

        self.contours_label_actor = None
        self.remove_old_label_actor = None
        self.faults_label_actor = None
        self.depth_label_actor = None
        self.play_label_actor = None

        self.elev_scalar_bar = None
        self.mag_scalar_bar = None

        self.slider_rep = None
        self.slider_widget = None

        self.contours_checkbox_rep = None
        self.contours_checkbox_widget = None

        self.remove_old_checkbox_rep = None
        self.remove_old_checkbox_widget = None

        self.faults_checkbox_rep = None
        self.faults_checkbox_widget = None

        self.depth_checkbox_rep = None
        self.depth_checkbox_widget = None

        self.play_button_rep = None
        self.play_button_widget = None

        self.mag_min = float(self.df["mag"].min())
        self.mag_max = float(self.df["mag"].max())
        self.elev_min = float(np.nanmin(self.dem))
        self.elev_max = float(np.nanmax(self.dem))

        self.terrain_scene_min = self.elev_min * self.z_scale
        self.terrain_scene_max = self.elev_max * self.z_scale

        self.mag_lut = self._build_magnitude_lut()
        self.terrain_lut = self._build_terrain_lut()

    @staticmethod
    def _load_catalog(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        required = ["time", "latitude", "longitude", "depth", "mag"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Catalog missing required columns: {missing}")

        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        for col in ["latitude", "longitude", "depth", "mag"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=required).sort_values("time").reset_index(drop=True)

        main_idx = df["mag"].idxmax()
        mainshock_time = df.loc[main_idx, "time"]

        #day 0 starts at the mainshock so anything before that is filtered out
        df["days_since_start"] = (df["time"] - mainshock_time).dt.total_seconds() / 86400.0
        df = df[df["days_since_start"] >= 0].reset_index(drop=True)

        df["is_mainshock"] = False
        main_idx = df["mag"].idxmax()
        df.loc[main_idx, "is_mainshock"] = True

        return df

    @staticmethod
    def _load_dem(dem_path: Path):
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(float)
            bounds = src.bounds
            nodata = src.nodata

        if nodata is not None:
            dem[dem == nodata] = np.nan

        dem[dem < -9999] = np.nan
        return dem, bounds

    @staticmethod
    def _build_magnitude_lut():
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()

        for i in range(256):
            t = i / 255.0
            r = min(1.0, 0.25 + 0.75 * t)
            g = min(1.0, 0.05 + 0.85 * (t ** 1.5))
            b = max(0.0, 0.45 - 0.40 * t)
            lut.SetTableValue(i, r, g, b, 1.0)

        return lut

    @staticmethod
    def _build_terrain_lut():
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()

        for i in range(256):
            t = i / 255.0
            if t < 0.35:
                r = 0.15 + 0.20 * t
                g = 0.35 + 0.45 * t
                b = 0.12 + 0.08 * t
            elif t < 0.75:
                u = (t - 0.35) / 0.40
                r = 0.40 + 0.35 * u
                g = 0.55 - 0.10 * u
                b = 0.20 - 0.05 * u
            else:
                u = (t - 0.75) / 0.25
                r = 0.78 + 0.18 * u
                g = 0.72 + 0.20 * u
                b = 0.68 + 0.22 * u
            lut.SetTableValue(i, r, g, b, 1.0)

        return lut

    def _build_terrain_grid(self):
        nrows, ncols = self.dem.shape
        xmin = self.bounds.left
        xmax = self.bounds.right
        ymin = self.bounds.bottom
        ymax = self.bounds.top

        xs = np.linspace(xmin, xmax, ncols)
        ys = np.linspace(ymax, ymin, nrows)

        points = vtk.vtkPoints()
        elev_array = vtk.vtkFloatArray()
        elev_array.SetName("elevation")
        valid_min = float(np.nanmin(self.dem))

        for row in range(nrows):
            for col in range(ncols):
                z = self.dem[row, col]
                if not np.isfinite(z):
                    z = valid_min
                z_scene = float(z) * self.z_scale
                points.InsertNextPoint(float(xs[col]), float(ys[row]), z_scene)
                elev_array.InsertNextValue(float(z))

        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(ncols, nrows, 1)
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(elev_array)
        self.terrain_grid = grid

    def _sample_elevation(self, lon: float, lat: float) -> float:
        xmin = self.bounds.left
        xmax = self.bounds.right
        ymin = self.bounds.bottom
        ymax = self.bounds.top

        nrows, ncols = self.dem.shape
        col = int(round((lon - xmin) / (xmax - xmin) * (ncols - 1)))
        row = int(round((ymax - lat) / (ymax - ymin) * (nrows - 1)))

        col = int(np.clip(col, 0, ncols - 1))
        row = int(np.clip(row, 0, nrows - 1))

        z = self.dem[row, col]
        if not np.isfinite(z):
            z = np.nanmin(self.dem)

        return float(z) * self.z_scale

    def _event_z(self, lon: float, lat: float, depth: float) -> float:
        terrain_z = self._sample_elevation(lon, lat)

        if self.depth_mode:
            return terrain_z + self.depth_base_offset - (float(depth) * self.depth_z_scale)

        return terrain_z + self.point_lift

    def _mainshock_z(self) -> float:
        lon = float(self.mainshock["longitude"])
        lat = float(self.mainshock["latitude"])
        depth = float(self.mainshock["depth"])

        if self.depth_mode:
            return self._event_z(lon, lat, depth)

        return self._sample_elevation(lon, lat) + self.point_lift + 0.012

    def _subsurface_top_z(self) -> float:
        return self.terrain_scene_max + 0.015

    def _subsurface_bottom_z(self) -> float:
        deepest_event_floor = self.terrain_scene_min + self.depth_base_offset - (self.max_depth * self.depth_z_scale)
        return deepest_event_floor - 0.03

    def _build_subsurface_box_polydata(self) -> vtk.vtkPolyData:
        xmin = self.bounds.left
        xmax = self.bounds.right
        ymin = self.bounds.bottom
        ymax = self.bounds.top

        top_z = self._subsurface_top_z()
        bottom_z = self._subsurface_bottom_z()

        points = vtk.vtkPoints()
        points.InsertNextPoint(xmin, ymin, top_z)
        points.InsertNextPoint(xmax, ymin, top_z)
        points.InsertNextPoint(xmax, ymax, top_z)
        points.InsertNextPoint(xmin, ymax, top_z)
        points.InsertNextPoint(xmin, ymin, bottom_z)
        points.InsertNextPoint(xmax, ymin, bottom_z)
        points.InsertNextPoint(xmax, ymax, bottom_z)
        points.InsertNextPoint(xmin, ymax, bottom_z)

        polys = vtk.vtkCellArray()

        def add_quad(a, b, c, d):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, a)
            quad.GetPointIds().SetId(1, b)
            quad.GetPointIds().SetId(2, c)
            quad.GetPointIds().SetId(3, d)
            polys.InsertNextCell(quad)

        add_quad(4, 5, 6, 7)
        add_quad(0, 1, 5, 4)
        add_quad(1, 2, 6, 5)
        add_quad(2, 3, 7, 6)
        add_quad(3, 0, 4, 7)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(polys)
        return poly

    def _events_for_day(self, day: float):
        shown = self.df[self.df["days_since_start"] <= day]

        recent_start = max(0.0, day - self.recent_window_days)
        mid_start = max(0.0, day - 3.0 * self.recent_window_days)
        far_start = max(0.0, day - 6.0 * self.recent_window_days)

        recent = shown[shown["days_since_start"] >= recent_start]
        old_near = shown[
            (shown["days_since_start"] < recent_start)
            & (shown["days_since_start"] >= mid_start)
        ]
        old_mid = shown[
            (shown["days_since_start"] < mid_start)
            & (shown["days_since_start"] >= far_start)
        ]
        old_far = shown[shown["days_since_start"] < far_start]

        return shown, old_far, old_mid, old_near, recent

    def _make_event_polydata(self, events: pd.DataFrame, recent: bool) -> vtk.vtkPolyData:
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()

        mag_array = vtk.vtkFloatArray()
        mag_array.SetName("mag")

        scale_array = vtk.vtkFloatArray()
        scale_array.SetName("scale")

        mag_range = max(self.mag_max - self.mag_min, 1e-6)

        for _, row in events.iterrows():
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            mag = float(row["mag"])
            depth = float(row["depth"])

            z = self._event_z(lon, lat, depth)

            pid = points.InsertNextPoint(lon, lat, z)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)

            mag_array.InsertNextValue(mag)

            base_scale = (mag - self.mag_min) / mag_range
            if recent:
                radius_scale = 0.006 + 0.014 * base_scale
            else:
                radius_scale = 0.0048 + 0.0105 * base_scale

            scale_array.InsertNextValue(radius_scale)

        poly.SetPoints(points)
        poly.SetVerts(verts)
        poly.GetPointData().AddArray(mag_array)
        poly.GetPointData().AddArray(scale_array)
        poly.GetPointData().SetActiveScalars("mag")
        return poly

    def _setup_glyph_pipeline(self, opacity: float):
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(18)
        sphere.SetPhiResolution(18)
        sphere.SetRadius(1.0)

        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetSourceConnection(sphere.GetOutputPort())
        mapper.ScalingOn()
        mapper.SetScaleArray("scale")
        mapper.SetScaleModeToScaleByMagnitude()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("mag")
        mapper.SetLookupTable(self.mag_lut)
        mapper.SetScalarRange(self.mag_min, self.mag_max)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetAmbient(0.2)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.1)

        return mapper, actor

    @staticmethod
    def _gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
        if radius is None:
            radius = max(1, int(3 * sigma))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    def _gaussian_blur_2d(self, grid: np.ndarray, sigma: float) -> np.ndarray:
        kernel = self._gaussian_kernel_1d(sigma)
        blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, grid)
        blurred = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, blurred)
        return blurred

    def _build_contour_polydata(self, density_events: pd.DataFrame) -> vtk.vtkPolyData:
        output_poly = vtk.vtkPolyData()

        if len(density_events) < self.contour_min_events:
            return output_poly

        xmin = self.bounds.left
        xmax = self.bounds.right
        ymin = self.bounds.bottom
        ymax = self.bounds.top

        x = density_events["longitude"].to_numpy()
        y = density_events["latitude"].to_numpy()

        #this is the whole contour idea. i count where points are dense, smooth it, and then draw outlines
        H, _, _ = np.histogram2d(
            x,
            y,
            bins=[self.contour_bins, self.contour_bins],
            range=[[xmin, xmax], [ymin, ymax]]
        )

        H = H.T

        H = self._gaussian_blur_2d(H, self.contour_sigma)

        max_h = np.nanmax(H)
        if not np.isfinite(max_h) or max_h <= 0:
            return output_poly

        H_norm = H / max_h

        image = vtk.vtkImageData()
        image.SetDimensions(self.contour_bins, self.contour_bins, 1)
        image.SetOrigin(xmin, ymin, 0.0)
        image.SetSpacing(
            (xmax - xmin) / (self.contour_bins - 1),
            (ymax - ymin) / (self.contour_bins - 1),
            1.0
        )

        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetName("density_norm")

        for row in range(self.contour_bins):
            for col in range(self.contour_bins):
                scalars.InsertNextValue(float(H_norm[row, col]))

        image.GetPointData().SetScalars(scalars)

        contour = vtk.vtkContourFilter()
        contour.SetInputData(image)
        contour.GenerateValues(self.contour_levels, self.contour_scalar_min, self.contour_scalar_max)
        contour.Update()

        contour_poly = vtk.vtkPolyData()
        contour_poly.DeepCopy(contour.GetOutput())

        pts = contour_poly.GetPoints()
        if pts is None:
            return output_poly

        for i in range(pts.GetNumberOfPoints()):
            lon, lat, _ = pts.GetPoint(i)
            z = self._sample_elevation(lon, lat) + self.contour_lift
            pts.SetPoint(i, lon, lat, z)

        contour_poly.Modified()

        #i remove the scalar values here so VTK cannot sneak the rainbow colors back in
        #the contours are yellow on purpose because they stuck out and so color is not encoding a density value anymore
        contour_poly.GetPointData().SetScalars(None)
        contour_poly.GetPointData().RemoveArray("density_norm")
        contour_poly.GetCellData().SetScalars(None)
        contour_poly.GetCellData().RemoveArray("density_norm")

        output_poly.DeepCopy(contour_poly)
        return output_poly

    @staticmethod
    def _distance_2d(pt_a, pt_b):
        return ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** 0.5

    def _clean_fault_part(self, part_points):
        if len(part_points) < 2:
            return []

        cleaned = [part_points[0]]
        for pt in part_points[1:]:
            if self._distance_2d(pt, cleaned[-1]) > 1e-9:
                cleaned.append(pt)

        if len(cleaned) < 2:
            return []

        total_length = 0.0
        for i in range(1, len(cleaned)):
            total_length += self._distance_2d(cleaned[i - 1], cleaned[i])

        if total_length < self.fault_min_segment_length:
            return []

        return cleaned

    def _build_fault_polydata(self) -> vtk.vtkPolyData:
        poly = vtk.vtkPolyData()

        if not self.has_fault_data:
            return poly

        reader = shapefile.Reader(str(self.faults_path))
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        for shape_rec in reader.iterShapeRecords():
            shape = shape_rec.shape

            if shape.shapeType not in [3, 13, 23]:
                continue

            pts = shape.points
            if len(pts) < 2:
                continue

            parts = list(shape.parts) + [len(pts)]

            for pidx in range(len(parts) - 1):
                start = parts[pidx]
                end = parts[pidx + 1]

                raw_part = pts[start:end]
                part_points = self._clean_fault_part(raw_part)

                if len(part_points) < 2:
                    continue

                vtk_line = vtk.vtkPolyLine()
                vtk_line.GetPointIds().SetNumberOfIds(len(part_points))

                for i, (lon, lat) in enumerate(part_points):
                    z = self._sample_elevation(lon, lat) + self.fault_lift
                    pid = points.InsertNextPoint(float(lon), float(lat), float(z))
                    vtk_line.GetPointIds().SetId(i, pid)

                lines.InsertNextCell(vtk_line)

        poly.SetPoints(points)
        poly.SetLines(lines)
        return poly

    def _build_fault_curtain_polydata(self) -> vtk.vtkPolyData:
        poly = vtk.vtkPolyData()

        if not self.has_fault_data:
            return poly

        reader = shapefile.Reader(str(self.faults_path))
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()

        bottom_z = self._subsurface_bottom_z()

        for shape_rec in reader.iterShapeRecords():
            shape = shape_rec.shape

            if shape.shapeType not in [3, 13, 23]:
                continue

            pts = shape.points
            if len(pts) < 2:
                continue

            parts = list(shape.parts) + [len(pts)]

            for pidx in range(len(parts) - 1):
                start = parts[pidx]
                end = parts[pidx + 1]

                raw_part = pts[start:end]
                part_points = self._clean_fault_part(raw_part)

                if len(part_points) < 2:
                    continue

                for i in range(len(part_points) - 1):
                    lon0, lat0 = part_points[i]
                    lon1, lat1 = part_points[i + 1]

                    top_z0 = self._sample_elevation(lon0, lat0) + 0.002
                    top_z1 = self._sample_elevation(lon1, lat1) + 0.002

                    p0_top = points.InsertNextPoint(float(lon0), float(lat0), float(top_z0))
                    p1_top = points.InsertNextPoint(float(lon1), float(lat1), float(top_z1))
                    p1_bot = points.InsertNextPoint(float(lon1), float(lat1), float(bottom_z))
                    p0_bot = points.InsertNextPoint(float(lon0), float(lat0), float(bottom_z))

                    quad = vtk.vtkQuad()
                    quad.GetPointIds().SetId(0, p0_top)
                    quad.GetPointIds().SetId(1, p1_top)
                    quad.GetPointIds().SetId(2, p1_bot)
                    quad.GetPointIds().SetId(3, p0_bot)
                    polys.InsertNextCell(quad)

        poly.SetPoints(points)
        poly.SetPolys(polys)
        return poly

    def _make_checkbox_image(self, checked: bool) -> vtk.vtkImageData:
        size = 26
        canvas = vtk.vtkImageCanvasSource2D()
        canvas.SetScalarTypeToUnsignedChar()
        canvas.SetNumberOfScalarComponents(3)
        canvas.SetExtent(0, size - 1, 0, size - 1, 0, 0)

        canvas.SetDrawColor(255, 255, 255)
        canvas.FillBox(0, size - 1, 0, size - 1)

        canvas.SetDrawColor(20, 20, 20)
        canvas.FillBox(0, size - 1, 0, 1)
        canvas.FillBox(0, size - 1, size - 2, size - 1)
        canvas.FillBox(0, 1, 0, size - 1)
        canvas.FillBox(size - 2, size - 1, 0, size - 1)

        if checked:
            canvas.SetDrawColor(90, 235, 255)
            canvas.FillBox(3, size - 4, 3, size - 4)

            canvas.SetDrawColor(0, 0, 0)
            for i in range(5, size - 5):
                canvas.DrawSegment(i, i, i + 1, i + 1)
                canvas.DrawSegment(i, size - 1 - i, i + 1, size - 2 - i)
                canvas.DrawSegment(i, i + 1, i + 1, i)
                canvas.DrawSegment(i, size - 2 - i, i + 1, size - 1 - i)

        canvas.Update()
        return canvas.GetOutput()

    def _make_play_button_image(self, playing: bool) -> vtk.vtkImageData:
        size = 96
        canvas = vtk.vtkImageCanvasSource2D()
        canvas.SetScalarTypeToUnsignedChar()
        canvas.SetNumberOfScalarComponents(3)
        canvas.SetExtent(0, size - 1, 0, size - 1, 0, 0)

        canvas.SetDrawColor(255, 255, 255)
        canvas.FillBox(0, size - 1, 0, size - 1)

        canvas.SetDrawColor(40, 40, 40)
        canvas.FillBox(0, size - 1, 0, 2)
        canvas.FillBox(0, size - 1, size - 3, size - 1)
        canvas.FillBox(0, 2, 0, size - 1)
        canvas.FillBox(size - 3, size - 1, 0, size - 1)

        canvas.SetDrawColor(232, 232, 232)
        canvas.FillBox(3, size - 4, 3, size - 4)

        if not playing:
            canvas.SetDrawColor(20, 90, 240)
            left_x = 32
            right_x = 66
            center_y = 48

            for x in range(left_x, right_x + 1):
                progress = (x - left_x) / max(1, (right_x - left_x))
                half_height = int(20 * (1.0 - progress))
                y_min = center_y - half_height
                y_max = center_y + half_height
                if y_min <= y_max:
                    canvas.FillBox(x, x, y_min, y_max)
        else:
            canvas.SetDrawColor(220, 40, 40)
            canvas.FillBox(31, 40, 24, 72)
            canvas.FillBox(56, 65, 24, 72)

        canvas.Update()
        return canvas.GetOutput()

    def _style_scalar_bar(self, scalar_bar, title_text):
        scalar_bar.SetTitle(title_text)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.GetTitleTextProperty().SetColor(1.0, 1.0, 1.0)
        scalar_bar.GetLabelTextProperty().SetColor(1.0, 1.0, 1.0)
        scalar_bar.GetTitleTextProperty().SetBold(0)
        scalar_bar.GetTitleTextProperty().SetFontSize(10)
        scalar_bar.GetLabelTextProperty().SetFontSize(9)

        try:
            scalar_bar.UnconstrainedFontSizeOn()
        except Exception:
            pass

    def _setup_scene(self):
        #most of the vtk plumbing lives here. this is where i have my renderer, terrain, glyphs, contours, faults, text, and widgets
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)

        self.renderer.SetUseDepthPeeling(1)
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.1)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1500, 940)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetWindowName("Ridgecrest Aftershock Evolution (VTK + Location-Density Contours)")
        self.render_window.SetAlphaBitPlanes(1)
        self.render_window.SetMultiSamples(0)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        self.subsurface_polydata.DeepCopy(self._build_subsurface_box_polydata())

        subsurface_mapper = vtk.vtkPolyDataMapper()
        subsurface_mapper.SetInputData(self.subsurface_polydata)

        self.subsurface_actor = vtk.vtkActor()
        self.subsurface_actor.SetMapper(subsurface_mapper)
        self.subsurface_actor.GetProperty().SetColor(0.55, 0.38, 0.22)
        self.subsurface_actor.GetProperty().SetOpacity(0.24)
        self.subsurface_actor.GetProperty().SetAmbient(0.35)
        self.subsurface_actor.GetProperty().SetDiffuse(0.65)
        self.renderer.AddActor(self.subsurface_actor)

        edge_filter = vtk.vtkExtractEdges()
        edge_filter.SetInputData(self.subsurface_polydata)

        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputConnection(edge_filter.GetOutputPort())

        self.subsurface_outline_actor = vtk.vtkActor()
        self.subsurface_outline_actor.SetMapper(edge_mapper)
        self.subsurface_outline_actor.GetProperty().SetColor(0.35, 0.22, 0.10)
        self.subsurface_outline_actor.GetProperty().SetLineWidth(1.5)
        self.subsurface_outline_actor.GetProperty().SetOpacity(0.55)
        self.renderer.AddActor(self.subsurface_outline_actor)

        terrain_mapper = vtk.vtkDataSetMapper()
        terrain_mapper.SetInputData(self.terrain_grid)
        terrain_mapper.SetLookupTable(self.terrain_lut)
        terrain_mapper.SetScalarRange(self.elev_min, self.elev_max)

        self.terrain_actor = vtk.vtkActor()
        self.terrain_actor.SetMapper(terrain_mapper)
        self.terrain_actor.GetProperty().SetInterpolationToPhong()
        self.renderer.AddActor(self.terrain_actor)

        self.old_far_mapper, self.old_far_actor = self._setup_glyph_pipeline(self.old_far_opacity)
        self.old_mid_mapper, self.old_mid_actor = self._setup_glyph_pipeline(self.old_mid_opacity)
        self.old_near_mapper, self.old_near_actor = self._setup_glyph_pipeline(self.old_near_opacity)
        self.recent_mapper, self.recent_actor = self._setup_glyph_pipeline(self.recent_point_opacity)

        self.old_far_mapper.SetInputData(self.old_far_polydata)
        self.old_mid_mapper.SetInputData(self.old_mid_polydata)
        self.old_near_mapper.SetInputData(self.old_near_polydata)
        self.recent_mapper.SetInputData(self.recent_polydata)

        self.renderer.AddActor(self.old_far_actor)
        self.renderer.AddActor(self.old_mid_actor)
        self.renderer.AddActor(self.old_near_actor)
        self.renderer.AddActor(self.recent_actor)

        self.contour_halo_tube = vtk.vtkTubeFilter()
        self.contour_halo_tube.SetInputData(self.contour_polydata)
        self.contour_halo_tube.SetRadius(self.contour_halo_radius)
        self.contour_halo_tube.SetNumberOfSides(18)
        self.contour_halo_tube.CappingOn()

        contour_halo_mapper = vtk.vtkPolyDataMapper()
        contour_halo_mapper.SetInputConnection(self.contour_halo_tube.GetOutputPort())
        contour_halo_mapper.ScalarVisibilityOff()

        self.contour_halo_actor = vtk.vtkActor()
        self.contour_halo_actor.SetMapper(contour_halo_mapper)

        #i keep the white halo off because it was covering the yellow contours and making it look glitchy/difficult to read
        self.contour_halo_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.contour_halo_actor.GetProperty().SetOpacity(0.0)
        self.contour_halo_actor.SetVisibility(0)
        self.renderer.AddActor(self.contour_halo_actor)

        self.contour_color_tube = vtk.vtkTubeFilter()
        self.contour_color_tube.SetInputData(self.contour_polydata)
        self.contour_color_tube.SetRadius(self.contour_color_radius)
        self.contour_color_tube.SetNumberOfSides(18)
        self.contour_color_tube.CappingOn()

        self.contour_color_mapper = vtk.vtkPolyDataMapper()
        self.contour_color_mapper.SetInputConnection(self.contour_color_tube.GetOutputPort())

        self.contour_color_mapper.ScalarVisibilityOff()

        self.contour_color_actor = vtk.vtkActor()
        self.contour_color_actor.SetMapper(self.contour_color_mapper)

        self.contour_color_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        self.contour_color_actor.GetProperty().SetOpacity(1.0)
        self.contour_color_actor.GetProperty().SetAmbient(1.0)
        self.contour_color_actor.GetProperty().SetDiffuse(0.0)
        self.contour_color_actor.GetProperty().SetSpecular(0.0)

        self.renderer.AddActor(self.contour_color_actor)

        self.fault_polydata.DeepCopy(self._build_fault_polydata())

        self.fault_halo_tube = vtk.vtkTubeFilter()
        self.fault_halo_tube.SetInputData(self.fault_polydata)
        self.fault_halo_tube.SetRadius(self.fault_halo_radius)
        self.fault_halo_tube.SetNumberOfSides(18)
        self.fault_halo_tube.CappingOn()

        fault_halo_mapper = vtk.vtkPolyDataMapper()
        fault_halo_mapper.SetInputConnection(self.fault_halo_tube.GetOutputPort())
        fault_halo_mapper.ScalarVisibilityOff()

        self.fault_halo_actor = vtk.vtkActor()
        self.fault_halo_actor.SetMapper(fault_halo_mapper)
        self.fault_halo_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.fault_halo_actor.GetProperty().SetOpacity(0.75)
        self.renderer.AddActor(self.fault_halo_actor)

        self.fault_color_tube = vtk.vtkTubeFilter()
        self.fault_color_tube.SetInputData(self.fault_polydata)
        self.fault_color_tube.SetRadius(self.fault_color_radius)
        self.fault_color_tube.SetNumberOfSides(18)
        self.fault_color_tube.CappingOn()

        fault_color_mapper = vtk.vtkPolyDataMapper()
        fault_color_mapper.SetInputConnection(self.fault_color_tube.GetOutputPort())
        fault_color_mapper.ScalarVisibilityOff()

        self.fault_color_actor = vtk.vtkActor()
        self.fault_color_actor.SetMapper(fault_color_mapper)
        self.fault_color_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.fault_color_actor.GetProperty().SetOpacity(1.0)
        self.renderer.AddActor(self.fault_color_actor)

        #in depth mode the surface fault traces turn into these vertical curtain references
        self.fault_curtain_polydata.DeepCopy(self._build_fault_curtain_polydata())

        curtain_mapper = vtk.vtkPolyDataMapper()
        curtain_mapper.SetInputData(self.fault_curtain_polydata)

        self.fault_curtain_actor = vtk.vtkActor()
        self.fault_curtain_actor.SetMapper(curtain_mapper)
        self.fault_curtain_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.fault_curtain_actor.GetProperty().SetOpacity(0.18)
        self.fault_curtain_actor.GetProperty().SetAmbient(0.45)
        self.fault_curtain_actor.GetProperty().SetDiffuse(0.55)
        self.renderer.AddActor(self.fault_curtain_actor)

        curtain_edges = vtk.vtkExtractEdges()
        curtain_edges.SetInputData(self.fault_curtain_polydata)

        curtain_edge_mapper = vtk.vtkPolyDataMapper()
        curtain_edge_mapper.SetInputConnection(curtain_edges.GetOutputPort())

        self.fault_curtain_outline_actor = vtk.vtkActor()
        self.fault_curtain_outline_actor.SetMapper(curtain_edge_mapper)
        self.fault_curtain_outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.fault_curtain_outline_actor.GetProperty().SetOpacity(0.45)
        self.fault_curtain_outline_actor.GetProperty().SetLineWidth(1.2)
        self.renderer.AddActor(self.fault_curtain_outline_actor)

        self._add_mainshock_marker()

        #colorbars
        self.elev_scalar_bar = vtk.vtkScalarBarActor()
        self.elev_scalar_bar.SetLookupTable(self.terrain_lut)
        self.elev_scalar_bar.SetPosition(0.020, 0.19)
        self.elev_scalar_bar.SetWidth(0.040)
        self.elev_scalar_bar.SetHeight(0.35)
        self._style_scalar_bar(self.elev_scalar_bar, "Elevation")
        self.renderer.AddViewProp(self.elev_scalar_bar)

        self.mag_scalar_bar = vtk.vtkScalarBarActor()
        self.mag_scalar_bar.SetLookupTable(self.mag_lut)
        self.mag_scalar_bar.SetPosition(0.070, 0.19)
        self.mag_scalar_bar.SetWidth(0.040)
        self.mag_scalar_bar.SetHeight(0.35)
        self._style_scalar_bar(self.mag_scalar_bar, "Magnitude")
        self.renderer.AddViewProp(self.mag_scalar_bar)

        #text
        self.title_actor = vtk.vtkTextActor()
        self.title_actor.SetDisplayPosition(28, 860)
        self.title_actor.GetTextProperty().SetFontSize(18)
        self.title_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddViewProp(self.title_actor)

        self.info_actor = vtk.vtkTextActor()
        self.info_actor.SetDisplayPosition(980, 26)
        self.info_actor.GetTextProperty().SetFontSize(13)
        self.info_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.info_actor.GetTextProperty().SetBackgroundColor(0.93, 0.93, 0.93)
        self.info_actor.GetTextProperty().SetBackgroundOpacity(0.95)
        self.info_actor.GetTextProperty().SetFrame(1)
        self.info_actor.GetTextProperty().SetFrameColor(0.65, 0.65, 0.65)
        self.info_actor.GetTextProperty().SetLineSpacing(1.05)
        self.renderer.AddViewProp(self.info_actor)

        self.contours_label_actor = vtk.vtkTextActor()
        self.contours_label_actor.SetDisplayPosition(1036, 748)
        self.contours_label_actor.SetInput("Contours")
        self.contours_label_actor.GetTextProperty().SetFontSize(16)
        self.contours_label_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddViewProp(self.contours_label_actor)

        self.remove_old_label_actor = vtk.vtkTextActor()
        self.remove_old_label_actor.SetDisplayPosition(1036, 688)
        self.remove_old_label_actor.SetInput("Remove old")
        self.remove_old_label_actor.GetTextProperty().SetFontSize(16)
        self.remove_old_label_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddViewProp(self.remove_old_label_actor)

        self.faults_label_actor = vtk.vtkTextActor()
        self.faults_label_actor.SetDisplayPosition(1036, 628)
        self.faults_label_actor.SetInput("Fault traces")
        self.faults_label_actor.GetTextProperty().SetFontSize(16)
        self.faults_label_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddViewProp(self.faults_label_actor)

        self.depth_label_actor = vtk.vtkTextActor()
        self.depth_label_actor.SetDisplayPosition(1036, 568)
        self.depth_label_actor.SetInput("Depth mode")
        self.depth_label_actor.GetTextProperty().SetFontSize(16)
        self.depth_label_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddViewProp(self.depth_label_actor)

        self.play_label_actor = vtk.vtkTextActor()
        self.play_label_actor.SetDisplayPosition(1110, 474)
        self.play_label_actor.SetInput("Play / Pause")
        self.play_label_actor.GetTextProperty().SetFontSize(15)
        self.play_label_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.renderer.AddViewProp(self.play_label_actor)

        #widgets
        self._setup_slider()
        self._setup_contours_checkbox()
        self._setup_remove_old_checkbox()
        self._setup_faults_checkbox()
        self._setup_depth_checkbox()
        self._setup_play_button()

        self._update_depth_mode_visuals()

        self.renderer.ResetCamera()
        cam = self.renderer.GetActiveCamera()
        cam.Elevation(38)
        cam.Azimuth(-42)
        cam.Zoom(1.25)
        self.renderer.ResetCameraClippingRange()

    def _add_mainshock_marker(self):
        lon = float(self.mainshock["longitude"])
        lat = float(self.mainshock["latitude"])
        z = self._mainshock_z()

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(lon, lat, z)
        sphere.SetRadius(0.012)
        sphere.SetThetaResolution(26)
        sphere.SetPhiResolution(26)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        self.mainshock_actor = vtk.vtkActor()
        self.mainshock_actor.SetMapper(mapper)
        self.mainshock_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.renderer.AddActor(self.mainshock_actor)

    def _update_mainshock_actor(self):
        if self.mainshock_actor is None:
            return

        lon = float(self.mainshock["longitude"])
        lat = float(self.mainshock["latitude"])
        z = self._mainshock_z()

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(lon, lat, z)
        sphere.SetRadius(0.012)
        sphere.SetThetaResolution(26)
        sphere.SetPhiResolution(26)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        self.mainshock_actor.SetMapper(mapper)

    def _update_fault_visuals(self):
        show_surface_faults = self.show_faults and self.has_fault_data and (not self.depth_mode)
        show_fault_curtains = self.show_faults and self.has_fault_data and self.depth_mode

        self.fault_halo_actor.SetVisibility(1 if show_surface_faults else 0)
        self.fault_color_actor.SetVisibility(1 if show_surface_faults else 0)

        self.fault_curtain_actor.SetVisibility(1 if show_fault_curtains else 0)
        self.fault_curtain_outline_actor.SetVisibility(1 if show_fault_curtains else 0)

    def _update_depth_mode_visuals(self):
        if self.depth_mode:
            self.terrain_actor.GetProperty().SetOpacity(self.depth_terrain_opacity)
            self.subsurface_actor.SetVisibility(1)
            self.subsurface_outline_actor.SetVisibility(1)
        else:
            self.terrain_actor.GetProperty().SetOpacity(1.0)
            self.subsurface_actor.SetVisibility(0)
            self.subsurface_outline_actor.SetVisibility(0)

        self._update_fault_visuals()
        self.renderer.ResetCameraClippingRange()

    def _setup_slider(self):
        self.slider_rep = vtk.vtkSliderRepresentation2D()
        self.slider_rep.SetMinimumValue(0.0)
        self.slider_rep.SetMaximumValue(self.max_days)
        self.slider_rep.SetValue(0.0)
        self.slider_rep.SetTitleText("Day")

        self.slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        self.slider_rep.GetPoint1Coordinate().SetValue(0.18, 0.14)
        self.slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        self.slider_rep.GetPoint2Coordinate().SetValue(0.56, 0.14)

        self.slider_rep.SetSliderLength(0.03)
        self.slider_rep.SetSliderWidth(0.04)
        self.slider_rep.SetTubeWidth(0.01)
        self.slider_rep.SetLabelHeight(0.02)
        self.slider_rep.SetTitleHeight(0.025)

        self.slider_rep.GetTubeProperty().SetColor(0.35, 0.35, 0.35)
        self.slider_rep.GetSliderProperty().SetColor(0.10, 0.35, 0.95)
        self.slider_rep.GetCapProperty().SetColor(0.25, 0.25, 0.25)
        self.slider_rep.GetTitleProperty().SetColor(0.0, 0.0, 0.0)
        self.slider_rep.GetLabelProperty().SetColor(0.0, 0.0, 0.0)
        self.slider_rep.GetSelectedProperty().SetColor(0.95, 0.20, 0.20)

        self.slider_widget = vtk.vtkSliderWidget()
        self.slider_widget.SetInteractor(self.interactor)
        self.slider_widget.SetRepresentation(self.slider_rep)
        self.slider_widget.SetAnimationModeToAnimate()
        self.slider_widget.EnabledOn()

        def slider_callback(caller, _event):
            self.current_day = float(caller.GetRepresentation().GetValue())
            self._update_scene(self.current_day)

        self.slider_widget.AddObserver(vtk.vtkCommand.InteractionEvent, slider_callback)

    def _setup_contours_checkbox(self):
        self.contours_checkbox_rep = vtk.vtkTexturedButtonRepresentation2D()
        self.contours_checkbox_rep.SetNumberOfStates(2)
        self.contours_checkbox_rep.SetButtonTexture(0, self._make_checkbox_image(False))
        self.contours_checkbox_rep.SetButtonTexture(1, self._make_checkbox_image(True))
        self.contours_checkbox_rep.SetState(1 if self.show_contours else 0)
        self.contours_checkbox_rep.PlaceWidget([1000, 1028, 744, 772, 0, 0])

        self.contours_checkbox_widget = vtk.vtkButtonWidget()
        self.contours_checkbox_widget.SetInteractor(self.interactor)
        self.contours_checkbox_widget.SetRepresentation(self.contours_checkbox_rep)
        self.contours_checkbox_widget.EnabledOn()

        def button_callback(widget, _event):
            self.show_contours = bool(widget.GetRepresentation().GetState())

            self.contour_halo_actor.SetVisibility(0)

            self.contour_color_actor.SetVisibility(1 if self.show_contours else 0)

            self.render_window.Render()

        self.contours_checkbox_widget.AddObserver(vtk.vtkCommand.StateChangedEvent, button_callback)

    def _setup_remove_old_checkbox(self):
        self.remove_old_checkbox_rep = vtk.vtkTexturedButtonRepresentation2D()
        self.remove_old_checkbox_rep.SetNumberOfStates(2)
        self.remove_old_checkbox_rep.SetButtonTexture(0, self._make_checkbox_image(False))
        self.remove_old_checkbox_rep.SetButtonTexture(1, self._make_checkbox_image(True))
        self.remove_old_checkbox_rep.SetState(1 if self.remove_old_shocks else 0)
        self.remove_old_checkbox_rep.PlaceWidget([1000, 1028, 684, 712, 0, 0])

        self.remove_old_checkbox_widget = vtk.vtkButtonWidget()
        self.remove_old_checkbox_widget.SetInteractor(self.interactor)
        self.remove_old_checkbox_widget.SetRepresentation(self.remove_old_checkbox_rep)
        self.remove_old_checkbox_widget.EnabledOn()

        def button_callback(widget, _event):
            self.remove_old_shocks = bool(widget.GetRepresentation().GetState())
            self._apply_old_visibility()
            self.render_window.Render()

        self.remove_old_checkbox_widget.AddObserver(vtk.vtkCommand.StateChangedEvent, button_callback)

    def _setup_faults_checkbox(self):
        self.faults_checkbox_rep = vtk.vtkTexturedButtonRepresentation2D()
        self.faults_checkbox_rep.SetNumberOfStates(2)
        self.faults_checkbox_rep.SetButtonTexture(0, self._make_checkbox_image(False))
        self.faults_checkbox_rep.SetButtonTexture(1, self._make_checkbox_image(True))
        self.faults_checkbox_rep.SetState(1 if self.show_faults else 0)
        self.faults_checkbox_rep.PlaceWidget([1000, 1028, 624, 652, 0, 0])

        self.faults_checkbox_widget = vtk.vtkButtonWidget()
        self.faults_checkbox_widget.SetInteractor(self.interactor)
        self.faults_checkbox_widget.SetRepresentation(self.faults_checkbox_rep)
        self.faults_checkbox_widget.EnabledOn()

        def button_callback(widget, _event):
            self.show_faults = bool(widget.GetRepresentation().GetState()) and self.has_fault_data
            self._update_fault_visuals()
            self.render_window.Render()

        self.faults_checkbox_widget.AddObserver(vtk.vtkCommand.StateChangedEvent, button_callback)

    def _setup_depth_checkbox(self):
        self.depth_checkbox_rep = vtk.vtkTexturedButtonRepresentation2D()
        self.depth_checkbox_rep.SetNumberOfStates(2)
        self.depth_checkbox_rep.SetButtonTexture(0, self._make_checkbox_image(False))
        self.depth_checkbox_rep.SetButtonTexture(1, self._make_checkbox_image(True))
        self.depth_checkbox_rep.SetState(1 if self.depth_mode else 0)
        self.depth_checkbox_rep.PlaceWidget([1000, 1028, 564, 592, 0, 0])

        self.depth_checkbox_widget = vtk.vtkButtonWidget()
        self.depth_checkbox_widget.SetInteractor(self.interactor)
        self.depth_checkbox_widget.SetRepresentation(self.depth_checkbox_rep)
        self.depth_checkbox_widget.EnabledOn()

        def button_callback(widget, _event):
            self.depth_mode = bool(widget.GetRepresentation().GetState())
            self._update_depth_mode_visuals()
            self._update_mainshock_actor()
            self._update_scene(self.current_day)
            self.render_window.Render()

        self.depth_checkbox_widget.AddObserver(vtk.vtkCommand.StateChangedEvent, button_callback)

    def _setup_play_button(self):
        self.play_button_rep = vtk.vtkTexturedButtonRepresentation2D()
        self.play_button_rep.SetNumberOfStates(2)
        self.play_button_rep.SetButtonTexture(0, self._make_play_button_image(False))
        self.play_button_rep.SetButtonTexture(1, self._make_play_button_image(True))
        self.play_button_rep.SetState(1 if self.playing else 0)
        self.play_button_rep.PlaceWidget([1000, 1102, 450, 550, 0, 0])

        self.play_button_widget = vtk.vtkButtonWidget()
        self.play_button_widget.SetInteractor(self.interactor)
        self.play_button_widget.SetRepresentation(self.play_button_rep)
        self.play_button_widget.EnabledOn()

        def button_callback(widget, _event):
            self.playing = bool(widget.GetRepresentation().GetState())
            self.render_window.Render()

        self.play_button_widget.AddObserver(vtk.vtkCommand.StateChangedEvent, button_callback)

    def _apply_old_visibility(self):
        if self.remove_old_shocks:
            self.old_far_actor.SetVisibility(0)
            self.old_mid_actor.SetVisibility(0)
            self.old_near_actor.SetVisibility(0)
        else:
            self.old_far_actor.SetVisibility(1)
            self.old_mid_actor.SetVisibility(1)
            self.old_near_actor.SetVisibility(1)
            self.old_far_actor.GetProperty().SetOpacity(self.old_far_opacity)
            self.old_mid_actor.GetProperty().SetOpacity(self.old_mid_opacity)
            self.old_near_actor.GetProperty().SetOpacity(self.old_near_opacity)

    def _update_text(self, day: float, count: int):
        faults_text = "Fault traces: on" if (self.show_faults and self.has_fault_data) else "Fault traces: off"

        if self.depth_mode:
            depth_text = "Depth mode: subsurface view"
            fault_text = "Faults become vertical curtains"
        else:
            depth_text = "Depth mode: surface view"
            fault_text = "Faults shown as surface traces"

        info_text = (
            "INFORMATION\n"
            f"Mainshock: M {self.mainshock['mag']:.1f}\n"
            "Time range:\n"
            f"{self.df['time'].min().date()} to {self.df['time'].max().date()}\n"
            "Spheres: size + color = magnitude\n"
            "Yellow contours: location-density clustering\n"
            "Nested lines = stronger spatial clustering\n"
            "Contours do not encode magnitude or shaking\n"
            f"{faults_text}\n"
            f"{fault_text}\n"
            f"{depth_text}\n"
            "Depth on = translucent terrain\n"
            "and subsurface earth box\n"
            "Remove old checked = hide old shocks\n"
            "Unchecked = fade older shocks\n"
            "Mouse: rotate / zoom / pan\n"
            "Space: play / pause"
        )

        self.title_actor.SetInput(
            f"Ridgecrest Aftershock Evolution (3D)\n"
            f"Day {day:.2f} | Showing {count} events"
        )
        self.info_actor.SetInput(info_text)

    def _update_scene(self, day: float):
        #this function is the live update loop. the slider/playback changes end up here
        shown_events, old_far_events, old_mid_events, old_near_events, recent_events = self._events_for_day(day)

        self.old_far_polydata.DeepCopy(self._make_event_polydata(old_far_events, recent=False))
        self.old_mid_polydata.DeepCopy(self._make_event_polydata(old_mid_events, recent=False))
        self.old_near_polydata.DeepCopy(self._make_event_polydata(old_near_events, recent=False))
        self.recent_polydata.DeepCopy(self._make_event_polydata(recent_events, recent=True))

        self.old_far_mapper.SetInputData(self.old_far_polydata)
        self.old_mid_mapper.SetInputData(self.old_mid_polydata)
        self.old_near_mapper.SetInputData(self.old_near_polydata)
        self.recent_mapper.SetInputData(self.recent_polydata)

        self._update_mainshock_actor()

        #for contours i use all aftershocks shown so far but i do not include the mainshock itself
        density_events = shown_events[~shown_events["is_mainshock"]].copy()
        contour_poly = self._build_contour_polydata(density_events)
        self.contour_polydata.DeepCopy(contour_poly)

        self.contour_halo_tube.SetInputData(self.contour_polydata)
        self.contour_halo_tube.Update()

        self.contour_color_tube.SetInputData(self.contour_polydata)
        self.contour_color_tube.Update()

        self.contour_halo_actor.SetVisibility(0)
        self.contour_color_actor.SetVisibility(1 if self.show_contours else 0)

        self._update_depth_mode_visuals()
        self._apply_old_visibility()
        self._update_text(day, len(shown_events))

        self.render_window.Render()

    def _on_key_press(self, obj, _event):
        key = obj.GetKeySym()
        if key == "space":
            self.playing = not self.playing
            self.play_button_rep.SetState(1 if self.playing else 0)
            self.render_window.Render()

    def _on_timer(self, _obj, _event):
        if not self.playing:
            return

        self.current_day += self.play_step_days
        if self.current_day >= self.max_days:
            self.current_day = self.max_days
            self.playing = False
            self.play_button_rep.SetState(0)

        self.slider_rep.SetValue(self.current_day)
        self._update_scene(self.current_day)

    def run(self):
        self._build_terrain_grid()
        self._setup_scene()
        self._update_scene(0.0)

        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self._on_key_press)
        self.interactor.AddObserver(vtk.vtkCommand.TimerEvent, self._on_timer)

        self.interactor.Initialize()
        self.interactor.CreateRepeatingTimer(self.timer_ms)
        self.render_window.Render()
        self.interactor.Start()


def main():
    parser = argparse.ArgumentParser(description="VTK 3D Ridgecrest aftershock viewer with yellow location-density contours")
    parser.add_argument(
        "-i",
        "--input",
        default="ridgecrest_aftershocks_cleaned.csv",
        help="Cleaned earthquake CSV",
    )
    parser.add_argument(
        "--dem",
        default="ridgecrest_dem.tiff",
        help="DEM GeoTIFF",
    )
    parser.add_argument(
        "--faults",
        default=None,
        help="Path to extracted Ridgecrest fault-rupture shapefile (.shp)",
    )
    args = parser.parse_args()

    faults_path = Path(args.faults) if args.faults else None
    viewer = RidgecrestVTKViewer(Path(args.input), Path(args.dem), faults_path)
    viewer.run()


if __name__ == "__main__":
    main()
