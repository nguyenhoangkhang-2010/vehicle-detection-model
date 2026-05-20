import numpy as np

def check_intersection(A, B, C, D):
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p2[0] - p1[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw (A, B, C) != ccw(A, B, D)

def get_direction(line_start, line_end, track_start, track_end):
    v_line = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
    v_move = np.array([track_end[0] - track_start[0], track_end[1] - track_start[1]])
    
    cross_product = np.cross(v_line, v_move)
    return 1 if cross_product >= 0 else -1