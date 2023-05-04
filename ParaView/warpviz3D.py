# trace generated using paraview version 5.11.0
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
data_3_000000vtu = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# get display properties
data_3_000000vtuDisplay = GetDisplayProperties(data_3_000000vtu, view=renderView1)

# set scalar coloring
ColorBy(data_3_000000vtuDisplay, ('CELLS', 'u', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
data_3_000000vtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
data_3_000000vtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'u'
uLUT = GetColorTransferFunction('u')

# get opacity transfer function/opacity map for 'u'
uPWF = GetOpacityTransferFunction('u')

# get 2D transfer function for 'u'
uTF2D = GetTransferFunction2D('u')

# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=data_3_000000vtu)
cellDatatoPointData1.CellDataArraytoprocess = ['cell_id', 'grid_dim', 'is_mortar', 'mortar_side', 'specific_volume', 'subdomain_id', 'u']

# show data in view
cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
cellDatatoPointData1Display.Representation = 'Surface'
cellDatatoPointData1Display.ColorArrayName = [None, '']
cellDatatoPointData1Display.SelectTCoordArray = 'None'
cellDatatoPointData1Display.SelectNormalArray = 'None'
cellDatatoPointData1Display.SelectTangentArray = 'None'
cellDatatoPointData1Display.OSPRayScaleArray = 'cell_id'
cellDatatoPointData1Display.OSPRayScaleFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.SelectOrientationVectors = 'None'
cellDatatoPointData1Display.ScaleFactor = 0.30000000000000004
cellDatatoPointData1Display.SelectScaleArray = 'None'
cellDatatoPointData1Display.GlyphType = 'Arrow'
cellDatatoPointData1Display.GlyphTableIndexArray = 'None'
cellDatatoPointData1Display.GaussianRadius = 0.015
cellDatatoPointData1Display.SetScaleArray = ['POINTS', 'cell_id']
cellDatatoPointData1Display.ScaleTransferFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.OpacityArray = ['POINTS', 'cell_id']
cellDatatoPointData1Display.OpacityTransferFunction = 'PiecewiseFunction'
cellDatatoPointData1Display.DataAxesGrid = 'GridAxesRepresentation'
cellDatatoPointData1Display.PolarAxes = 'PolarAxesRepresentation'
cellDatatoPointData1Display.ScalarOpacityUnitDistance = 0.26388782292607027
cellDatatoPointData1Display.OpacityArrayName = ['POINTS', 'cell_id']
cellDatatoPointData1Display.SelectInputVectors = ['POINTS', 'u']
cellDatatoPointData1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
cellDatatoPointData1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1499.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
cellDatatoPointData1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1499.0, 1.0, 0.5, 0.0]

# hide data in view
Hide(data_3_000000vtu, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on cellDatatoPointData1Display
cellDatatoPointData1Display.Opacity = 0.8

# set scalar coloring
ColorBy(cellDatatoPointData1Display, ('POINTS', 'u', 'Magnitude'))

# rescale color and/or opacity maps used to include current data range
cellDatatoPointData1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
cellDatatoPointData1Display.SetScalarBarVisibility(renderView1, True)

# create a new 'Warp By Vector'
warpByVector1 = WarpByVector(registrationName='WarpByVector1', Input=cellDatatoPointData1)
warpByVector1.Vectors = ['POINTS', 'u']

# Properties modified on warpByVector1
warpByVector1.ScaleFactor = 30.0

# show data in view
warpByVector1Display = Show(warpByVector1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
warpByVector1Display.Representation = 'Surface'
warpByVector1Display.ColorArrayName = ['POINTS', 'u']
warpByVector1Display.LookupTable = uLUT
warpByVector1Display.SelectTCoordArray = 'None'
warpByVector1Display.SelectNormalArray = 'None'
warpByVector1Display.SelectTangentArray = 'None'
warpByVector1Display.OSPRayScaleArray = 'cell_id'
warpByVector1Display.OSPRayScaleFunction = 'PiecewiseFunction'
warpByVector1Display.SelectOrientationVectors = 'None'
warpByVector1Display.ScaleFactor = 0.30000000000000004
warpByVector1Display.SelectScaleArray = 'None'
warpByVector1Display.GlyphType = 'Arrow'
warpByVector1Display.GlyphTableIndexArray = 'None'
warpByVector1Display.GaussianRadius = 0.015
warpByVector1Display.SetScaleArray = ['POINTS', 'cell_id']
warpByVector1Display.ScaleTransferFunction = 'PiecewiseFunction'
warpByVector1Display.OpacityArray = ['POINTS', 'cell_id']
warpByVector1Display.OpacityTransferFunction = 'PiecewiseFunction'
warpByVector1Display.DataAxesGrid = 'GridAxesRepresentation'
warpByVector1Display.PolarAxes = 'PolarAxesRepresentation'
warpByVector1Display.ScalarOpacityFunction = uPWF
warpByVector1Display.ScalarOpacityUnitDistance = 0.26388782292607027
warpByVector1Display.OpacityArrayName = ['POINTS', 'cell_id']
warpByVector1Display.SelectInputVectors = ['POINTS', 'u']
warpByVector1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
warpByVector1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1499.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
warpByVector1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1499.0, 1.0, 0.5, 0.0]

# hide data in view
Hide(cellDatatoPointData1, renderView1)

# show color bar/color legend
warpByVector1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(data_3_000000vtu)

# change representation type
data_3_000000vtuDisplay.SetRepresentationType('Outline')

# set active source
SetActiveSource(data_3_000000vtu)

# show data in view
data_3_000000vtuDisplay = Show(data_3_000000vtu, renderView1, 'UnstructuredGridRepresentation')

# show color bar/color legend
data_3_000000vtuDisplay.SetScalarBarVisibility(renderView1, True)

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

# set active source
SetActiveSource(cellDatatoPointData1)

# set active source
SetActiveSource(data_3_000000vtu)

# change representation type
data_3_000000vtuDisplay.SetRepresentationType('Feature Edges')

# change representation type
data_3_000000vtuDisplay.SetRepresentationType('Outline')

# Properties modified on renderView1
renderView1.Background = [0.6705882352941176, 0.6705882352941176, 0.6705882352941176]

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 1

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1279, 718)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-3.2859880281341014, -1.5605881899094853, -3.4878830901066857]
renderView1.CameraFocalPoint = [0.12499999999999914, 1.4999999999999973, 0.12499999999999972]
renderView1.CameraViewUp = [-0.8112196652970323, 0.3621730756168996, 0.4590787709463013]
renderView1.CameraParallelScale = 1.5103807466993215

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).