from hexasphere import hexgrid, projection

my_grid = hexgrid.HexGrid()
my_projection = projection.Projection(my_grid)
my_grid.projection = my_projection