# Ridgecrest Aftershock Visualization — Run Instructions

This repository contains the code and data needed to run my CS 530 final project: an interactive 3D visualization of the 2019 Ridgecrest, California aftershock sequence.

The visualization uses:

- a cleaned Ridgecrest aftershock CSV file
- a DEM terrain GeoTIFF
- Ridgecrest fault trace shapefile data
- Python + VTK for rendering

This README focuses only on how to set up and run the project with the fault lines included.

---

## 1. Expected Project Folder Structure

Before running the code, make sure your repository is organized like this:

```text
CS530_Final_Project/
│
├── README.md
├── ridgecrest_viewer.py
│
└── data/
    ├── ridgecrest_aftershocks_cleaned.csv
    ├── ridgecrest_dem.tiff
    ├── Surface_Rupture_Ridgecrest_Prov_Rel_1.shp
    ├── Surface_Rupture_Ridgecrest_Prov_Rel_1.shx
    ├── Surface_Rupture_Ridgecrest_Prov_Rel_1.dbf
    ├── Surface_Rupture_Ridgecrest_Prov_Rel_1.prj
    └── any other shapefile sidecar files
```

The important part is that the `data` folder must be in the same folder as `ridgecrest_viewer.py`.

---

## 2. Important Note About the Fault Shapefile

The fault lines are loaded from a shapefile.

A shapefile is not just one file. The `.shp` file depends on several sidecar files with the same base name.

At minimum, the following files should be together in the `data` folder:

```text
Surface_Rupture_Ridgecrest_Prov_Rel_1.shp
Surface_Rupture_Ridgecrest_Prov_Rel_1.shx
Surface_Rupture_Ridgecrest_Prov_Rel_1.dbf
```

It is also recommended to include:

```text
Surface_Rupture_Ridgecrest_Prov_Rel_1.prj
```

and any other files that came with the shapefile download.

If only the `.shp` file is included and the `.shx` or `.dbf` files are missing, the fault traces may not load correctly.

---

## 3. Required Python Packages

The project requires the following Python packages:

```text
numpy
pandas
rasterio
pyshp
vtk
```

Install them with:

```bash
pip install numpy pandas rasterio pyshp vtk
```

---

## 4. Recommended Setup With a Virtual Environment

Using a virtual environment is recommended so the project dependencies do not conflict with other Python projects.

### Windows PowerShell

From the root project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas rasterio pyshp vtk
```

### macOS / Linux

From the root project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas rasterio pyshp vtk
```

After activation, the terminal prompt should show something like `(.venv)`.

---

## 5. Run the Visualization With Fault Lines Included

From the root project folder, run:

```bash
python ridgecrest_viewer.py --input data/ridgecrest_aftershocks_cleaned.csv --dem data/ridgecrest_dem.tiff --faults data/Surface_Rupture_Ridgecrest_Prov_Rel_1.shp
```

This command tells the program:

* `--input` is the cleaned earthquake catalog
* `--dem` is the terrain elevation file
* `--faults` is the Ridgecrest fault trace shapefile

The fault traces should appear as white lines in the surface view.

---

## 6. Expected Result

After running the command, a VTK window should open.

You should see:

* 3D terrain of the Ridgecrest region
* earthquake events plotted as spheres
* a red marker for the mainshock
* white mapped fault traces
* yellow location-density contour lines
* a time slider
* checkboxes for contours, old events, fault traces, and depth mode
* a play/pause button

If the fault lines are loaded correctly, the white fault traces should appear on top of the terrain when the `Fault traces` checkbox is selected.

---

## 7. Controls

| Control                              | What it does                                          |
| ------------------------------------ | ----------------------------------------------------- |
| Left mouse drag                      | Rotate the 3D scene                                   |
| Mouse wheel                          | Zoom in and out                                       |
| Middle mouse drag / right mouse drag | Pan the scene, depending on mouse settings            |
| Day slider                           | Move through the aftershock sequence                  |
| Play / Pause button                  | Animate the aftershock sequence                       |
| Spacebar                             | Toggle play/pause                                     |
| Contours checkbox                    | Show or hide yellow location-density contours         |
| Remove old checkbox                  | Hide older aftershocks instead of showing them faded  |
| Fault traces checkbox                | Show or hide the mapped fault traces                  |
| Depth mode checkbox                  | Switch between surface view and subsurface depth view |

---

## 8. What the Visualization Shows

### Earthquake Spheres

Each earthquake is shown as a sphere.

| Data field           | Visual encoding                   |
| -------------------- | --------------------------------- |
| Latitude / longitude | Sphere position                   |
| Magnitude            | Sphere size                       |
| Magnitude            | Sphere color                      |
| Event age            | Sphere opacity                    |
| Depth                | Subsurface position in depth mode |

Recent events are shown more clearly. Older events fade over time to reduce clutter.

### Fault Lines

The white lines show mapped Ridgecrest surface fault traces.

In surface mode, these appear as white lines on top of the terrain.

In depth mode, the fault traces become vertical translucent curtains so the user can compare the surface fault geometry with the earthquake locations below the surface.

### Yellow Contour Lines

The yellow contours show location-density clustering.

They are based only on earthquake longitude and latitude.

They are intended to show the general shape of where aftershocks are spatially clustered.

---

## 9. Main Run Command

In conclusion, the main command to run this project is:

```bash
python ridgecrest_viewer.py --input data/ridgecrest_aftershocks_cleaned.csv --dem data/ridgecrest_dem.tiff --faults data/Surface_Rupture_Ridgecrest_Prov_Rel_1.shp
```

```
```
