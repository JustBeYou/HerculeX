import numpy as np
from . import geometry
from gym.envs.classic_control import rendering

class BoardViewer:
    def __init__(self, board):
        screen_width = 600
        screen_height = 400
        hexagon_length = 15
        hexagon_border = 1
        
        n = board.shape[0]
        board_height =  n * hexagon_length * np.sqrt(3)
        board_width =  n * hexagon_length * np.sqrt(3) + n * hexagon_length / 2 * np.sqrt(3)
        
        left_margin = (screen_width - board_width) / 2
        top_margin = (screen_height * 1.1 - board_height) / 2
        
        x_align = left_margin
        y_align = screen_height - top_margin
                
        self.viewer = rendering.Viewer(screen_width, screen_height)
        
        x, y, length = x_align, y_align, hexagon_length
        
        hexagon = geometry.compute_hexagon(x, y, length)
        hexagons = [hexagon]
        displacement = length * np.sqrt(3)
        for _ in range(n - 1):
            hexagons.append(geometry.translate_points(hexagons[-1], displacement, 0))
        
        x_coef, y_coef = geometry.compute_hexagon_below_coefs(length)
        x_center_row, y_center_row = x, y
        
        self.cached_polygons = []
        
        for r, row in enumerate(board):
            x_center_col, y_center_col = x_center_row, y_center_row
            self.cached_polygons.append([])
            for q, (cell, poly) in enumerate(zip(row, hexagons)):
                self.cached_polygons[-1].append(poly)
                polygon_obj = rendering.PolyLine(poly, True)
                self.viewer.add_geom(polygon_obj)
                
                x_center_col += displacement
                
            x_center_row += x_coef
            y_center_row += y_coef
            hexagons = [geometry.translate_points(h, x_coef, y_coef) for h in hexagons]
                    
    def color_hexagon(self, r, q, color):
        polygon_obj = rendering.FilledPolygon(self.cached_polygons[r][q])
        polygon_obj.set_color(*color)
        self.viewer.geoms.insert(0, polygon_obj)
